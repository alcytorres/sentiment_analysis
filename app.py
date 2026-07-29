# RAG + sentiment app for user-supplied financial text
# - Paste text or upload a .txt file → chunk → embed → Neo4j vector index
# - Retrieve relevant chunks via LangChain + Neo4j (with a similarity floor)
# - Show the retrieved evidence itself, numbered and cited
# - Score sentiment with FinBERT (per snippet + overall)

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os, logging, glob
from typing import List, Dict, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from neo4j import GraphDatabase

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

# ---------------------------
# App & logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.secret_key = "dev-only-secret"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MAX_CHARS = 50_000

# Retrieval tuning
TOP_K = 10  # candidates pulled from the vector index
SHOW_K = 3  # snippets shown as the answer
# Similarity floor (0-1) on Neo4j's cosine score. Below this a question is
# treated as unrelated to the indexed text, so no evidence and no sentiment are
# shown. On-topic questions score ~0.67+; off-topic ones sit below ~0.6.
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "0.65"))

# ---------------------------
# Neo4j configuration
# ---------------------------
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise RuntimeError(
        "Set the NEO4J_PASSWORD environment variable before starting the app."
    )
NEO4J_INDEX = "financial_rag"

# ---------------------------
# Embeddings (small, fast)
# ---------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------
# FinBERT sentiment (local)
# ---------------------------
FINBERT = "ProsusAI/finbert"
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT)

# Read the class order from the model config instead of assuming it. FinBERT
# outputs [positive, negative, neutral], which is not the usual ordering.
LABEL_INDEX = {
    label.lower(): idx for idx, label in finbert_model.config.id2label.items()
}


def finbert_scalar(text: str) -> float:
    """Return a scalar in [-1, 1] where +1 is very bullish and -1 is very bearish."""
    try:
        enc = finbert_tokenizer(
            text, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            out = finbert_model(**enc)
            probs = F.softmax(out.logits[0], dim=-1).tolist()
        return float(probs[LABEL_INDEX["positive"]] - probs[LABEL_INDEX["negative"]])
    except Exception as e:
        logging.warning("FinBERT failed: %s", e)
        return 0.0


def to_label_and_score(avg_scalar: float) -> Dict[str, str]:
    """Map scalar [-1,1] to label, 0-100% bullishness, and a UI tone class."""
    pct = (avg_scalar + 1.0) / 2.0 * 100.0
    if pct >= 80:
        lab, tone = "Very Bullish", "bullish"
    elif pct >= 60:
        lab, tone = "Bullish", "bullish"
    elif pct >= 40:
        lab, tone = "Neutral", "neutral"
    elif pct >= 20:
        lab, tone = "Bearish", "bearish"
    else:
        lab, tone = "Very Bearish", "bearish"
    return {"label": lab, "score_pct": f"{pct:.1f}", "tone": tone}


# ---------------------------
# Helpers
# ---------------------------
def split_docs(docs: List[Document]) -> List[Document]:
    """Split documents into RAG-sized chunks."""
    # Small chunks keep each snippet on one topic, which sharpens retrieval and
    # keeps the quoted evidence short enough to read in the UI.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def clear_existing_chunks() -> None:
    """Delete all Chunk nodes so the next index run is a clean overwrite."""
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n:Chunk) DETACH DELETE n")
    driver.close()
    logging.info("Cleared existing Chunk nodes from Neo4j.")


def ensure_index_exists(documents: List[Document]) -> None:
    """Create/overwrite the Neo4j vector index from documents."""
    Neo4jVector.from_documents(
        documents=documents,
        embedding=embeddings,
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name=NEO4J_INDEX,
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    logging.info("Neo4j vector index created/refreshed: %s", NEO4J_INDEX)


def load_vectorstore_existing() -> Neo4jVector:
    """Open an existing index. Raise helpful error if missing."""
    try:
        return Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name=NEO4J_INDEX,
            text_node_property="text",
            embedding_node_property="embedding",
        )
    except Exception as e:
        raise RuntimeError(
            f"Vector index '{NEO4J_INDEX}' not found in Neo4j. "
            f"Index some text first. Details: {e}"
        )


def _fingerprint(text: str) -> str:
    """Normalized key used to drop near-duplicate chunks caused by overlap."""
    return " ".join(text.lower().split())[:160]


def build_evidence(scored_docs: List[Tuple[Document, float]]) -> List[Dict]:
    """Turn scored chunks into numbered, deduplicated evidence for the UI."""
    evidence: List[Dict] = []
    seen: set = set()

    for doc, score in scored_docs:
        if score < RELEVANCE_MIN:
            continue

        text = " ".join((doc.page_content or "").split())
        if not text:
            continue

        key = _fingerprint(text)
        if key in seen:
            continue
        seen.add(key)

        scalar = finbert_scalar(text)
        sentiment = to_label_and_score(scalar)
        evidence.append(
            {
                "text": text,
                "source": doc.metadata.get("source", "unknown"),
                "match_pct": f"{score * 100:.0f}",
                "scalar": scalar,
                "label": sentiment["label"],
                "score_pct": sentiment["score_pct"],
                "tone": sentiment["tone"],
            }
        )

        if len(evidence) == SHOW_K:
            break

    return evidence


# ---------------------------
# Sample data
# ---------------------------
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")


def load_sample_texts() -> List[Document]:
    """Load bundled sample .txt files as Documents."""
    docs: List[Document] = []
    for path in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path)},
                )
            )
    return docs


# ---------------------------
# Indexing flow
# ---------------------------
def index_text(paste: str, file_text: str, file_name: str) -> str:
    """
    Build Documents from pasted text and/or uploaded .txt content,
    split → embed → overwrite Neo4j vector index.
    """
    documents: List[Document] = []

    if paste.strip():
        documents.append(
            Document(page_content=paste.strip(), metadata={"source": "pasted text"})
        )

    if file_text.strip():
        documents.append(
            Document(
                page_content=file_text.strip(),
                metadata={"source": file_name or "uploaded.txt"},
            )
        )

    if not documents:
        return "Please paste some text or upload a .txt file."

    total_chars = sum(len(d.page_content) for d in documents)
    if total_chars > MAX_CHARS:
        return f"Text too long ({total_chars:,} chars). Maximum is {MAX_CHARS:,} characters."

    chunks = split_docs(documents)
    if not chunks:
        return "No usable text found after splitting."

    clear_existing_chunks()
    ensure_index_exists(chunks)
    return f"Indexed {len(chunks)} chunks from {len(documents)} source(s)."


# ---------------------------
# Query flow (RAG + sentiment)
# ---------------------------
def query_rag(question: str) -> Dict:
    """Retrieve the best-matching chunks and score the sentiment of that evidence.

    Questions that no indexed chunk answers well enough return empty evidence so
    the UI can say so instead of citing unrelated text.
    """
    vs = load_vectorstore_existing()
    scored_docs = vs.similarity_search_with_score(question, k=TOP_K)
    evidence = build_evidence(scored_docs)

    if not evidence:
        return {"question": question, "evidence": [], "sentiment": None}

    scalars = [e["scalar"] for e in evidence]
    sentiment = to_label_and_score(sum(scalars) / len(scalars))

    # Averaging bullish and bearish evidence lands near zero, which would read as
    # "Neutral" and hide the disagreement.
    if max(scalars) > 0.2 and min(scalars) < -0.2:
        sentiment["label"] = "Mixed"
        sentiment["tone"] = "neutral"

    sentiment["sources"] = ", ".join(
        dict.fromkeys(e["source"] for e in evidence)  # unique, order preserved
    )

    return {"question": question, "evidence": evidence, "sentiment": sentiment}


# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "index":
            paste = request.form.get("corpus_text", "")
            file_text = ""
            file_name = ""
            f = request.files.get("file")
            if f and f.filename:
                if not f.filename.lower().endswith(".txt"):
                    flash("Only .txt files are accepted.", "danger")
                    return redirect(url_for("index"))
                try:
                    file_text = f.read().decode("utf-8")
                    file_name = secure_filename(f.filename)
                except UnicodeDecodeError:
                    flash("File must be valid UTF-8 text.", "danger")
                    return redirect(url_for("index"))
            try:
                msg = index_text(paste, file_text, file_name)
                flash(msg, "success")
            except Exception as e:
                logging.exception("Indexing failed")
                flash(f"Indexing failed: {e}", "danger")
            return redirect(url_for("index"))

        if action == "load_sample":
            try:
                sample_docs = load_sample_texts()
                if not sample_docs:
                    flash("No sample files found in samples/ directory.", "danger")
                    return redirect(url_for("index"))
                chunks = split_docs(sample_docs)
                clear_existing_chunks()
                ensure_index_exists(chunks)
                flash(
                    f"Loaded sample data: {len(chunks)} chunks from "
                    f"{len(sample_docs)} file(s).",
                    "success",
                )
            except Exception as e:
                logging.exception("Loading sample data failed")
                flash(f"Sample loading failed: {e}", "danger")
            return redirect(url_for("index"))

        if "question" in request.form:
            q = (request.form.get("question") or "").strip()
            if not q:
                return render_template("index.html", message="Please enter a question.")
            try:
                result = query_rag(q)
                return render_template("index.html", result=result)
            except Exception as e:
                logging.exception("Query failed")
                return render_template("index.html", message=f"Query failed: {e}")

    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

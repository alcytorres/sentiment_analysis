
# NEW: Hardened RAG + sentiment app for Tesla earnings PDFs
# - Index 4 PDFs into Neo4j (vector index)
# - Retrieve relevant chunks via LangChain + Neo4j
# - Score sentiment with FinBERT (scalar + label)
# - Write a short, cited answer with FLAN-T5 (local, CPU)
# - No OCR/Unstructured deps; pure PyPDF

# NEW: Add flash + PRG so the page shows a visible success/failure banner after indexing
from flask import Flask, render_template, request, redirect, url_for, flash  # NEW
from werkzeug.utils import secure_filename
import os, logging
from typing import List, Dict, Tuple

# LangChain (stable)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings            # FIX: no deprecation warning
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# Transformers (local models)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,      # FinBERT
    AutoModelForSeq2SeqLM,                   # FLAN-T5
    pipeline,
)

# ---------------------------
# App & logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.secret_key = "dev-only-secret"  # NEW: needed for flash()
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # NEW: 50MB upload cap
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# Neo4j configuration
# ---------------------------
# Use bolt://localhost to avoid 127.0.1/403 quirks
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "AI-NEO4J-word4")
NEO4J_INDEX = "tesla_earnings"

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
finbert = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer, device=-1)

def finbert_scalar(text: str) -> float:
    """
    NEW: Return a scalar in [-1, 1] where +1 is very bullish and -1 is very bearish.
    """
    try:
        res = finbert(text[:512])[0]  # keep under 512 tokens/chars for safety
        lbl, score = res["label"].lower(), float(res["score"])
        if lbl == "positive":
            return score              # + (0..1)
        if lbl == "negative":
            return -score             # - (0..-1)
        return 0.0
    except Exception as e:
        logging.warning(f"FinBERT failed: {e}")
        return 0.0

def to_label_and_score(avg_scalar: float) -> Dict[str, str]:
    """
    Map scalar [-1,1] to label and 0-100% bullishness.
    """
    pct = (avg_scalar + 1.0) / 2.0 * 100.0
    if pct >= 80:
        lab = "Very Bullish"
    elif pct >= 60:
        lab = "Bullish"
    elif pct >= 40:
        lab = "Neutral"
    elif pct >= 20:
        lab = "Bearish"
    else:
        lab = "Very Bearish"
    return {"label": lab, "score_pct": f"{pct:.1f}"}

# ---------------------------
# Local LLM for answer writing (CPU-friendly)
# ---------------------------
GEN_MODEL = "google/flan-t5-small"  # tiny, instruction-following
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen = pipeline("text2text-generation", model=gen_model, tokenizer=gen_tokenizer, device=-1)

def write_cited_answer(question: str, numbered_ctx: List[str]) -> str:
    """
    NEW: Ask FLAN-T5 to answer using ONLY provided snippets [1]..[k] and include citations.
    """
    if not numbered_ctx:
        return "No relevant evidence was found in the indexed reports. Please re-index the PDFs and try again."
    context_block = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(numbered_ctx)])
    prompt = (
        "You are an equity research analyst. Using ONLY the context snippets below, "
        "answer the user's question in 3-6 sentences. Cite evidence with [n] where n is the snippet number. "
        "Do not invent facts. Be concise.\n\n"
        f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer:"
    )
    out = gen(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"].strip()
    return out

# ---------------------------
# Helpers
# ---------------------------
def split_docs(docs) -> List:
    """
    Split documents into RAG-sized chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)

def ensure_index_exists(documents: List) -> None:
    """
    Create/overwrite the Neo4j vector index from documents.
    """
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
    """
    Open an existing index. Raise helpful error if missing.
    """
    try:
        vs = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name=NEO4J_INDEX,
            text_node_property="text",
            embedding_node_property="embedding",
        )
        return vs
    except Exception as e:
        raise RuntimeError(
            f"Vector index '{NEO4J_INDEX}' not found in Neo4j. "
            f"Upload & index the 4 PDFs first. Details: {e}"
        )

def summarize_docs_for_llm(docs: List) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Prepare short snippets for the LLM + a separate list for UI evidence.
    """
    snippets: List[str] = []
    evidences: List[Dict[str, str]] = []
    for d in docs:
        text = (d.page_content or "").strip().replace("\n", " ")
        short = text[:450]
        src = os.path.basename(d.metadata.get("source", "")) or "unknown.pdf"
        page = d.metadata.get("page", "?")
        snippets.append(short)
        evidences.append({"source": src, "page": str(page), "snippet": short})
    return snippets, evidences

# ---------------------------
# Indexing flow
# ---------------------------
def index_pdfs(file_storages) -> str:
    """
    Save exactly 4 PDFs under ./data/, load → split → create Neo4j vector index.
    """
    if len(file_storages) != 4:
        return "Please upload exactly 4 PDF files."

    documents = []
    for fs in file_storages:
        if not fs or not fs.filename.lower().endswith(".pdf"):
            return "Only PDF files are allowed."
        fname = secure_filename(fs.filename)
        path = os.path.join(DATA_DIR, fname)
        fs.save(path)

        loader = PyPDFLoader(path)     # simple, reliable PDF text loader
        page_docs = loader.load()      # per-page docs with metadata (source, page)
        documents.extend(split_docs(page_docs))

    if not documents:
        return "No text extracted from PDFs."

    ensure_index_exists(documents)
    return "Indexed 4 PDFs successfully."

# ---------------------------
# Query flow (RAG + sentiment)
# ---------------------------
def query_rag(question: str) -> Dict[str, str]:
    """
    Retrieve top-k chunks, compute sentiment, and write a cited answer.
    """
    vs = load_vectorstore_existing()
    # Retrieve the most relevant chunks
    top_docs = vs.similarity_search(question, k=6)
    if not top_docs:
        return {"question": question, "answer": "No relevant evidence found. Try re-indexing the PDFs."}

    # Sentiment per chunk and overall
    scalars = [finbert_scalar(d.page_content) for d in top_docs]
    avg_scalar = sum(scalars) / max(len(scalars), 1)
    sent_meta = to_label_and_score(avg_scalar)

    # Cited answer
    snippets, evidences = summarize_docs_for_llm(top_docs)
    answer = write_cited_answer(question, snippets)

    # Prefix final sentiment & append human-readable citations
    cites = "; ".join([f"{e['source']} p.{e['page']}" for e in evidences])
    final = (
        f"Overall sentiment: {sent_meta['label']} ({sent_meta['score_pct']}%).\n"
        f"{answer}\n\nSources: {cites}"
    )
    return {"question": question, "answer": final}

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Upload/index
        # NEW: Upload/index with PRG + flash
        if "files" in request.files:
            files = request.files.getlist("files")
            try:
                msg = index_pdfs(files)
                flash(msg, "success")  # NEW
            except Exception as e:
                logging.exception("Indexing failed")
                flash(f"Indexing failed: {e}", "danger")  # NEW
            return redirect(url_for("index"))  # NEW: PRG

        # Ask a question
        if "question" in request.form:
            q = (request.form.get("question") or "").strip()
            if not q:
                return render_template("index.html", message="Please enter a question.")
            try:
                result = query_rag(q)
                return render_template("index.html", question=result["question"], answer=result["answer"])
            except Exception as e:
                logging.exception("Query failed")
                return render_template("index.html", message=f"Query failed: {e}")

    # GET
    return render_template("index.html")

# Health endpoint (optional)
@app.route("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    # Explicit host/port so you see the server start message
    app.run(host="127.0.0.1", port=5000, debug=True)
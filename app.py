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

# NEW: for robust FinBERT scoring without >512 warnings
import torch                                   # NEW
import torch.nn.functional as F                # NEW

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
# Keep a simple pipeline around for any future use; main scoring uses the NEW function below
finbert = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer, device=-1)

def finbert_scalar(text: str) -> float:
    """
    NEW: Return a scalar in [-1, 1] where +1 is very bullish and -1 is very bearish.
         Truncates safely and maps FinBERT logits to a stable scalar.
    """
    try:
        enc = finbert_tokenizer(
            text,
            truncation=True,       # NEW
            max_length=512,        # NEW
            return_tensors="pt"    # NEW
        )
        with torch.no_grad():      # NEW
            out = finbert_model(**enc)
            probs = F.softmax(out.logits[0], dim=-1).tolist()  # NEW
        # FinBERT label order: ['negative', 'neutral', 'positive']
        neg, neu, pos = probs
        return float(pos - neg)    # NEW: map to [-1, 1]
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
# GEN_MODEL = "google/flan-t5-small"  # tiny, instruction-following
# NEW: use base for better instruction following and paragraph writing
GEN_MODEL = "google/flan-t5-base"  # NEW
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen = pipeline("text2text-generation", model=gen_model, tokenizer=gen_tokenizer, device=-1)

def write_cited_answer(question: str, numbered_ctx: List[str]) -> str:
    """
    NEW: Ask FLAN-T5 to answer using ONLY provided snippets [1]..[k] and include citations.
         Trim context and force multi-sentence decoding so we don’t get a one-word answer.
    """
    if not numbered_ctx:
        return "No relevant evidence was found in the indexed reports. Please re-index the PDFs and try again."

    # Keep context tight so the model attends well
    trimmed = [c[:350] for c in numbered_ctx[:3]]
    context_block = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(trimmed)])

    prompt = (
        "You are an equity research analyst. Using ONLY the context snippets below, "
        "write a 3–6 sentence answer. Cite evidence with [n] where n is the snippet number. "
        "Do not invent facts. Do not answer with a single word; write full sentences. Be concise but complete.\n\n"
        f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer:"
    )

    # NOTE: removed return_full_text (not supported for text2text-generation)
    out = gen(
        prompt,
        max_new_tokens=220,
        min_new_tokens=60,
        num_beams=4,
        do_sample=False,
        clean_up_tokenization_spaces=True,
    )[0]["generated_text"].strip()

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
def query_rag(question: str):
    """
    Retrieve relevant chunks, write a cited answer with FLAN-T5, and append sentiment + sources.
    """
    # Load existing index (raises a helpful error if missing)
    vs = load_vectorstore_existing()
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    # 1) Retrieve
    # docs = retriever.get_relevant_documents(question)  # deprecated
    docs = retriever.invoke(question)  # NEW: modern API, same outcome (List[Document])

    if not docs:
        return {
            "question": question,
            "answer": "I couldn't find relevant context in the indexed PDFs. Try re-indexing or asking a broader question."
        }

    # 2) Prep snippets for LLM + evidence for UI
    snippets, evidences = summarize_docs_for_llm(docs)  # returns short strings + [{source,page,snippet}]

    # 3) Write cited answer using ONLY the retrieved context
    cited_answer = write_cited_answer(question, snippets)

    # 4) Sentiment on each retrieved chunk, then average → label + pct
    scalars = [finbert_scalar(d.page_content) for d in docs]  # NEW: internal truncation handles length
    avg_scalar = sum(scalars) / max(len(scalars), 1)
    sent = to_label_and_score(avg_scalar)  # {'label': ..., 'score_pct': 'xx.x'}

    # 5) Build sources string
    srcs = "; ".join(
        f"{os.path.basename(ev['source'])} p.{ev['page']}" for ev in evidences
    )

    # 6) Final composed answer (first the written, cited answer)
    final_text = (
        f"{cited_answer}\n\n"
        f"Overall sentiment: {sent['label']} ({sent['score_pct']}%).\n"
        f"Sources: {srcs}"
    )

    return {"question": question, "answer": final_text}

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

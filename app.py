# Import required libraries for Flask app, sentiment analysis, and text processing
from flask import Flask, render_template, request
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json
import re
import torch
import spacy  # Lightweight library for Named Entity Recognition (NER) to identify topics/companies
import numpy as np
from nltk.tokenize import sent_tokenize  # Splits text into sentences for contrast detection
import nltk
# NEW: Imports for LangChain, RAG, Neo4j, embeddings, PDF
from langchain_community.document_loaders import PyPDFLoader  # NEW: Load PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # NEW: Chunk text
from langchain_community.embeddings import HuggingFaceEmbeddings  # NEW: Embeddings model
from langchain_community.vectorstores.neo4j_vector import Neo4jVector  # NEW: Neo4j vector store
from langchain.chains import RetrievalQA  # NEW: RAG chain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline  # NEW: Local pipeline for LLM
from neo4j import GraphDatabase  # NEW: Neo4j driver
import os  # NEW: For file handling

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Initialize Flask app and models for sentiment analysis and NER
app = Flask(__name__)
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
nlp = spacy.load("en_core_web_sm")  # spaCy model for NER, identifies entities like "Tesla" or "Erythritol"

# NEW: Neo4j connection details (update with your Neo4j credentials)
NEO4J_URI = "neo4j://127.0.0.1:7687" # From Neo4j setup
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "AI-NEO4J-word4"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))  # NEW: Connect to Neo4j

# NEW: Embedding model for RAG
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Local model

# NEW: Local LLM for RAG using transformers pipeline
llm_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english"),
    device=-1  # CPU, change to 0 for GPU if available
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)  # NEW: Wrap pipeline for LangChain

# Define templates for dynamic sentiment explanations
templates = {
    "Very Bullish": "The article emphasizes {entity}'s exceptional performance and growth potential, with positive language like {phrases}{contrast}, driving a highly optimistic tone.",
    "Bullish": "The article highlights {entity}'s solid performance, with terms like {phrases}{contrast}, indicating a positive outlook.",
    "Neutral": "The article discusses {entity} with balanced language, including terms like {phrases}{contrast}, suggesting a neutral stance.",
    "Bearish": "The article points to challenges for {entity}, with negative terms like {phrases}{contrast}, reflecting a bearish sentiment.",
    "Very Bearish": "The article underscores significant issues for {entity}, with strong negative language like {phrases}{contrast}, indicating a highly bearish tone."
}

# Extract the main topic or company using spaCy NER
# Prioritizes ORG (companies), then PROPN/NOUN (e.g., "Erythritol", "Bitcoin"), defaults to "the topic"
def extract_entity(text):
    doc = nlp(text)
    # Try to find organizations first
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    if entities:
        return entities[0]
    # Fallback to proper nouns or nouns (e.g., "Erythritol", "Bitcoin")
    candidates = [token.text for token in doc if token.pos_ in ["PROPN", "NOUN"] and len(token.text) > 2]
    return candidates[0] if candidates else "the topic"

# Extract key phrases using FinBERT attention weights, with filtering
# Identifies influential words/phrases, skips punctuation, prefers multi-word phrases
def extract_key_phrases(text, tokens, attention, top_k=3):
    attention = attention.mean(dim=1).mean(dim=0)  # Average attention across heads/layers
    top_indices = attention.argsort(descending=True)[:top_k * 3]  # Get more candidates
    key_phrases = []
    # Convert tokens to words, filter out punctuation and special tokens
    for idx in top_indices:
        token = tokenizer.convert_ids_to_tokens([tokens[idx]])[0]
        if token not in ['[CLS]', '[SEP]'] and token.isalnum():  # Skip special tokens and punctuation
            key_phrases.append(token)
    # Try to form multi-word phrases by combining adjacent tokens
    words = text.lower().split()
    phrases = []
    for i in range(len(words) - 1):
        phrase = " ".join(words[i:i+2])
        if any(word in phrase for word in key_phrases):
            phrases.append(phrase)
    return phrases[:top_k] if phrases else key_phrases[:top_k] or ["performance", "market"]

# Detect contrasting sentiments in sentences
# Checks for mixed positive/negative sentiments to add context (e.g., "Tesla thrives, industry struggles")
def detect_contrast(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return ""
    results = sentiment_pipeline(sentences)
    labels = [r['label'].capitalize() for r in results]
    if "Positive" in labels and "Negative" in labels:
        return ", contrasted with challenges in the broader industry"
    return ""

# Analyze sentiment and generate enhanced explanation with intuitive labels
def analyze_sentiment(text):
    # Tokenize for attention weights
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_attention_mask=True)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1]  # Last layer attentions
    # Get sentiment label and raw score
    result = sentiment_pipeline(text)[0]
    label = result['label'].capitalize()
    raw_score = result['score']  # FinBERT confidence score (0 to 1)
    # Map to intuitive labels and normalized scores
    if label == 'Positive':
        normalized_score = raw_score * 100  # Scale to 60-100%
        if normalized_score >= 80:
            final_label = "Very Bullish"
        else:
            final_label = "Bullish"
    elif label == 'Negative':
        normalized_score = (1 - raw_score) * 100  # Invert to 0-39%
        if normalized_score <= 20:
            final_label = "Very Bearish"
        else:
            final_label = "Bearish"
    else:
        normalized_score = 50 - (raw_score * 10)  # Neutral around 40-59%
        final_label = "Neutral"
    # Approximate pos, neu, neg for chart
    if final_label in ["Very Bullish", "Bullish"]:
        pos = normalized_score / 100
        neg = 0
        neu = 1 - pos
    elif final_label in ["Very Bearish", "Bearish"]:
        neg = (100 - normalized_score) / 100
        pos = 0
        neu = 1 - neg
    else:
        neu = normalized_score / 100
        pos = 0
        neg = 0
    # Extract entity, phrases, and contrast
    entity = extract_entity(text)
    key_phrases = extract_key_phrases(text, inputs['input_ids'][0], attention[0], top_k=3)
    contrast = detect_contrast(text)
    # Generate explanation
    explanation = templates[final_label].format(
        entity=entity,
        phrases=", ".join(f"'{phrase}'" for phrase in key_phrases),
        contrast=contrast
    )
    return {
        'label': final_label,
        'score': f"{normalized_score:.2f}%",
        'explanation': explanation,
        'pos': pos,
        'neu': neu,
        'neg': neg
    }

# NEW: Function to index PDFs in Neo4j
def index_pdfs(files):
    documents = []
    for file in files:
        file_path = os.path.join("data", file.filename)
        file.save(file_path)  # Save uploaded PDF
        loader = PyPDFLoader(file_path)  # Load PDF
        docs = loader.load()  # Extract text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Chunk text
        chunks = splitter.split_documents(docs)  # Split into chunks
        documents.extend(chunks)
    # NEW: Store chunks in Neo4j as vector store
    vector_store = Neo4jVector.from_documents(
        documents,
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="tesla_earnings"  # Index name
    )
    return "Indexed 4 PDFs successfully."

# NEW: Function to query RAG and get sentiment
def query_rag(question):
    # NEW: Set up RAG chain with LangChain
    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="tesla_earnings"
    )
    retriever = vector_store.as_retriever()  # Retrieve from Neo4j
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple chain
        retriever=retriever,
        return_source_documents=True  # For citations
    )
    response = qa_chain({"query": question})  # Run RAG
    retrieved_docs = response['source_documents']  # Get evidence
    # NEW: Score sentiment on retrieved chunks
    sentiments = [analyze_sentiment(doc.page_content) for doc in retrieved_docs]  # Use existing sentiment function
    avg_label = max(set([s['label'] for s in sentiments]), key=[s['label'] for s in sentiments].count)  # Simple majority label
    # NEW: Generate explanation with citations
    explanation = f"Sentiment: {avg_label}. Explanation: {response['result']}. Evidence: " + "; ".join([f"Page {doc.metadata['page']} in {doc.metadata['source']}" for doc in retrieved_docs])
    return {
        'question': question,
        'answer': explanation
    }

# Update Flask route for upload and query
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' in request.files:  # NEW: Handle PDF upload
            files = request.files.getlist('files')
            if len(files) == 4:
                message = index_pdfs(files)
                return render_template('index.html', message=message)
            else:
                return render_template('index.html', message="Upload exactly 4 PDFs.")
        elif 'question' in request.form:  # NEW: Handle question
            question = request.form['question']
            result = query_rag(question)
            return render_template('index.html', question=result['question'], answer=result['answer'])
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

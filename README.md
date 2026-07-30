# Financial Sentiment Analyzer

Paste financial text, ask a question, and get back the exact passages that answer it — each scored bullish or bearish.

![App demo](demo_sentiment.gif)

## What It Does

Financial Sentiment Analyzer is a local RAG (Retrieval-Augmented Generation) web app for reading financial text quickly. Paste an article, earnings call notes, or research snippets, and the app splits the text into passages, embeds them, and stores them in a Neo4j vector index.

Ask a question and it returns the closest-matching passages, numbered and cited with a match percentage, then scores each one with FinBERT for bullish/bearish sentiment. Answers are the source text itself, so nothing is paraphrased or invented. If no passage matches the question closely enough, the app says so instead of citing unrelated text.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Python, Flask |
| Vector store | Neo4j vector index |
| RAG pipeline | LangChain (chunking, embeddings, retrieval) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Sentiment | FinBERT (ProsusAI) via Transformers + PyTorch |
| Frontend | Jinja templates, Bootstrap, custom CSS |

All models run locally on CPU — no external API calls or keys.

## Features

- **Cited evidence, not guesses** — answers are the retrieved passages, numbered `[1] [2] [3]` with their source and match percentage
- **Relevance gate** — off-topic questions return "no relevant evidence found" instead of citing unrelated text
- **Financial sentiment** — FinBERT scores every passage, plus an overall Bullish / Neutral / Mixed / Bearish label
- **Flexible input** — paste text, attach a `.txt` file, or load the bundled sample financial texts
- **Local and private** — runs entirely on your machine, no API keys
- **Dark and light mode** — theme toggle in the top-right corner

## Getting Started

### Prerequisites

- Python 3.10+
- A running Neo4j instance (Neo4j Desktop is easiest locally)
- Internet connection on the first run, to download the models once

### Setup

1. Clone the repository and enter it:

```bash
git clone https://github.com/alcytorres/sentiment_analysis.git
cd sentiment_analysis
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Point the app at your Neo4j instance:

```bash
export NEO4J_PASSWORD="your-password"
```

5. Start the server:

```bash
python3 app.py
```

Open http://127.0.0.1:5000, paste financial text (or click **Load Sample Data**), click **Index Text**, then ask a question.

### Optional configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEO4J_PASSWORD` | — | Required. Your Neo4j password |
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j connection string |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `RELEVANCE_MIN` | `0.65` | Similarity floor (0–1). Raise it to be stricter about what counts as relevant evidence |

First run downloads FinBERT (~400 MB) and the MiniLM embedding model (~90 MB).

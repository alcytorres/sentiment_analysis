# Financial Sentiment Analyzer

# Description
  A RAG (Retrieval-Augmented Generation) web app that indexes user-supplied financial text into a Neo4j vector database, retrieves relevant context for questions, generates cited answers with a local LLM, and scores the sentiment (bullish / bearish) of the retrieved evidence using FinBERT.

  Paste financial articles, research notes, or earnings call snippets (or upload a .txt file), index them, then ask questions and get grounded answers with citations and sentiment analysis.

# Getting Started
  These instructions will get you a copy of the project up and running on your local machine.

# Prerequisites
  - Python 3.8+
  - pip (Python package manager)
  - Neo4j database (running locally via Neo4j Desktop or accessible remotely)
  - Internet connection for initial model downloads

# Technologies Used
  - Flask (Web Framework)
  - Neo4j (Vector Database for RAG)
  - LangChain (Document processing and vector store integration)
  - Transformers (FinBERT for sentiment analysis, FLAN-T5 for answer generation)
  - PyTorch (Model inference)
  - Sentence Transformers (Embeddings)
  - Bootstrap (UI framework)
  - Custom CSS (styling)

# Installation
  1. Clone the repository:
      git clone https://github.com/alcytorres/sentiment_analysis.git

  2. Navigate to the project directory:
      cd sentiment_analysis

  3. Create a virtual environment:
      python3 -m venv venv

  4. Activate the virtual environment:
      - Windows: venv\Scripts\activate
      - Mac/Linux: source venv/bin/activate

  5. Install dependencies:
      pip install -r requirements.txt

  6. Set up Neo4j:
      - Install Neo4j Desktop and create a local instance
      - Start the instance and note the bolt URI and password
      - Set environment variables:
          export NEO4J_PASSWORD="your-password"
          export NEO4J_URL="bolt://localhost:7687"   # optional, this is the default
          export NEO4J_USER="neo4j"                  # optional, this is the default

# Starting the Server
  From the project directory:
    python3 app.py

  The app will be available at http://127.0.0.1:5000

# Usage
  1. Index Text:
     - Paste financial text (articles, research notes, earnings call snippets) into the text area
     - Optionally attach a .txt file
     - Click "Index Text" to chunk, embed, and store in Neo4j
     - Or click "Load Sample Data" to use the bundled sample financial texts

  2. Ask Questions:
     - Type a question about the indexed text (e.g., "What is the sentiment on revenue growth?")
     - The app retrieves the most relevant chunks, generates a cited answer, and scores sentiment

  3. View Results:
     - Cited answer with [1], [2], etc. referencing the source chunks
     - Overall sentiment label (Very Bullish, Bullish, Neutral, Bearish, Very Bearish)
     - Sentiment score percentage
     - Source references

  4. Theme Toggle: Switch between dark mode (default) and light mode using the button in the top-right corner.

# Key Features
  - RAG Architecture: Vector-based retrieval from Neo4j for context-aware answers
  - Financial Sentiment: FinBERT provides nuanced bullish/bearish classification
  - Cited Answers: FLAN-T5 generates answers with source citations
  - Local AI Models: Runs entirely on CPU with no external API calls
  - Clean Overwrite: Each index action replaces the previous data for simplicity
  - Sample Data: Built-in sample financial texts for quick demos

# Environment Variables
  - NEO4J_PASSWORD (required): Your Neo4j instance password
  - NEO4J_URL (optional): Connection string (default: bolt://localhost:7687)
  - NEO4J_USER (optional): Neo4j username (default: neo4j)

# Model Downloads
  First run will download:
  - FinBERT (~400MB)
  - FLAN-T5-base (~250MB)
  - Sentence transformer embeddings (~90MB)
  Ensure a stable internet connection for initial setup.

# License
  This project is open source and available under the MIT License.

# Acknowledgments
  - ProsusAI for FinBERT model
  - Google for FLAN-T5 model
  - LangChain community for RAG tools
  - Neo4j for vector database support
  - Sentence Transformers for embeddings

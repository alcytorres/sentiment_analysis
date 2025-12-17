# README

# Company Sentiment Analyzer

# Description
  The Company Sentiment Analyzer is an AI-powered RAG (Retrieval-Augmented Generation) application designed to analyze company earnings reports and answer questions with sentiment insights. The app indexes multiple earnings PDFs into a Neo4j vector database, retrieves relevant context, and generates cited answers using local AI models.

  Built with a clean interface, featuring a dark mode, the app helps users quickly extract insights from earnings reports, understand sentiment trends, and get detailed answers with source citations. It combines the power of vector search, financial sentiment analysis, and generative AI to provide comprehensive earnings analysis.

# Getting Started
  These instructions will get you a copy of the project up and running on your local machine.

# Prerequisites
  Before you begin, ensure you have met the following requirements:
    - Python version: 3.8+
    - pip (Python package manager)
    - Neo4j database (running locally or accessible via connection string)
    - Internet connection for initial model downloads

# Technologies Used
  - Flask (Web Framework)
  - Neo4j (Vector Database for RAG)
  - LangChain (Document processing and vector store integration)
  - Transformers (FinBERT for sentiment analysis, FLAN-T5 for answer generation)
  - PyTorch (Model inference)
  - Sentence Transformers (Embeddings)
  - PyPDF (PDF text extraction)
  - Bootstrap (UI framework)
  - Custom CSS (styling)

# Backend Installation
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
      - Install and start Neo4j database
      - Update environment variables if needed (default: bolt://localhost:7687)
      - Default credentials: neo4j / AI-NEO4J-word4 (change in app.py or via environment variables)

# Starting the Flask Server
  From the sentiment_analysis directory, run:
    python3 app.py

  The app will be available at http://127.0.0.1:5000

# Usage
  1. Upload PDFs: Upload exactly 4 earnings report PDF files using the upload form. The app will extract text, split into chunks, and index them into Neo4j.

  2. Ask Questions: Enter questions about the earnings reports (e.g., "What's the sentiment toward revenue growth in Q2 earnings?"). The app will:
     - Retrieve relevant context from the indexed PDFs
     - Generate a cited answer using FLAN-T5
     - Analyze sentiment using FinBERT
     - Display results with source references

  3. View Results: Answers include:
     - Detailed response with citations [1], [2], etc.
     - Overall sentiment label (Very Bullish, Bullish, Neutral, Bearish, Very Bearish)
     - Sentiment score percentage
     - Source document references with page numbers

  4. Theme Toggle: Use the toggle button in the top-right corner to switch between dark mode (default) and light mode.

# Key Features
  - RAG Architecture: Vector-based retrieval from Neo4j for accurate, context-aware answers
  - Financial Sentiment Analysis: FinBERT model provides nuanced sentiment classification for financial text
  - Cited Answers: FLAN-T5 generates answers with source citations for transparency
  - PDF Processing: Robust text extraction from earnings PDFs using PyPDF
  - Premium UI: Modern, minimal design with dark mode support
  - Local AI Models: Runs entirely on CPU with local models (no external API calls)
  - Source Tracking: Every answer includes references to source documents and page numbers

# Additional Configuration
  - Environment Variables (optional):
    - NEO4J_URL: Neo4j connection string (default: bolt://localhost:7687)
    - NEO4J_USER: Neo4j username (default: neo4j)
    - NEO4J_PASSWORD: Neo4j password (default: AI-NEO4J-word4)
  
  - Model Downloads: First run will download:
    - FinBERT (~400MB)
    - FLAN-T5-base (~250MB)
    - Sentence transformer embeddings (~90MB)
    - Ensure stable internet connection for initial setup

  - File Storage: Uploaded PDFs are stored in the `data/` directory

# License
  This project is open source and available under the MIT License.

# Acknowledgments
  - ProsusAI for FinBERT model
  - Google for FLAN-T5 model
  - LangChain community for RAG tools
  - Neo4j for vector database support
  - Sentence Transformers for embeddings

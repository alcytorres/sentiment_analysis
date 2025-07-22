# README

# Sentiment Analysis

# Description
  This Sentiment Analysis AI Agent is a personalized tool designed to analyze the sentiment of financial articles or reviews, classifying them as Very Bullish, Bullish, Neutral, Bearish, or Very Bearish with concise explanations. It serves as a powerful resource for tracking market sentiment, allowing users to evaluate text inputs and visualize sentiment trends.

  Built with a clean and intuitive interface, the app helps users assess financial texts, identify key entities (e.g., companies), and evaluate investment decisions based on sentiment analysis. It’s more than just a tool—it’s a dynamic assistant for financial insight.

# Getting Started
  These instructions will get you a copy of the project up and running on your local machine.

# Prerequisites
  Before you begin, ensure you have met the following requirements:
    - Python version: 3.8+
    - pip (Python package manager)

# Technologies Used
  - Flask (Web Framework)
  - Transformers (for FinBERT sentiment analysis)
  - spaCy (for Named Entity Recognition)
  - NLTK (for sentence tokenization)
  - Chart.js (for sentiment distribution visualization)
  - Bootstrap (for UI styling)

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

# Starting the Flask Server
  From the sentiment_analysis directory, run:
    python app.py

# Usage
  - Analyze Text: Paste a single article or multiple articles (separated by three newlines) into the text box and click "Analyze" to see sentiment results.
  - View Results: Check the table for sentiment labels, scores, and explanations.
  - Sentiment Distribution: View the bar chart to see the overall sentiment breakdown.
  - Scoring Explanation: Refer to the table explaining score ranges and meanings.

# Key Features
  - Text Analysis: Classifies sentiment with FinBERT for financial nuance.
  - Detailed Explanations: Provides entity-specific insights and key phrases.
  - Sentiment Visualization: Displays a bar chart of sentiment distribution.
  - Batch Processing: Handles multiple texts with three-newline separation.

# Additional Configuration
  - Ensure internet access for initial model downloads (e.g., FinBERT, ~400MB).
  - No local storage; results are ephemeral.

# License
  This project is open source and available under the MIT License.

# Acknowledgments
  - ProsusAI for FinBERT model.
  - spaCy and NLTK communities for NLP tools.
  - Chart.js and Bootstrap for visualization and styling support.
  
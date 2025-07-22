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

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Initialize Flask app and models for sentiment analysis and NER
app = Flask(__name__)
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
nlp = spacy.load("en_core_web_sm")  # spaCy model for NER, identifies entities like "Tesla" or "Erythritol"

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

# Define Flask route for web app
@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ""  # Initialize input_text for GET requests
    if request.method == 'POST':
        input_text = request.form['text']
        # Split only on three or more consecutive newlines
        texts = re.split(r'\n\s*\n\s*\n\s*', input_text.strip())
        results = [analyze_sentiment(t.strip()) for t in texts if t.strip()]
        
        # Aggregate for chart if batch
        if len(results) > 1:
            avg_pos = sum(r['pos'] for r in results) / len(results)
            avg_neu = sum(r['neu'] for r in results) / len(results)
            avg_neg = sum(r['neg'] for r in results) / len(results)
            chart_data = json.dumps({
                'labels': ['Positive', 'Neutral', 'Negative'],
                'data': [avg_pos, avg_neu, avg_neg]
            })
        else:
            chart_data = json.dumps({
                'labels': ['Positive', 'Neutral', 'Negative'],
                'data': [results[0]['pos'], results[0]['neu'], results[0]['neg']]
            }) if results else None
        
        print(results)  # Debug: Print results to check pos, neu, neg values
        # In index() function, before return render_template
        print("Chart Data:", chart_data)
        return render_template('index.html', results=results, chart_data=chart_data, input_text=input_text)
    
    return render_template('index.html', results=None, chart_data=None, input_text=input_text)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
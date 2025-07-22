# Import required libraries
from flask import Flask, render_template, request
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json
import re
import torch
import spacy  # New: Lightweight library for Named Entity Recognition (NER) to identify companies/topics
import numpy as np
from nltk.tokenize import sent_tokenize  # New: Splits text into sentences for contrast detection
import nltk

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Initialize Flask app and models
app = Flask(__name__)
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
nlp = spacy.load("en_core_web_sm")  # New: spaCy model for NER, identifies companies like "Tesla"

# Predefined templates for dynamic explanations
templates = {
    "Strongly Positive": "The article emphasizes {entity}'s exceptional performance and growth potential, with positive language like {phrases}{contrast}, driving a highly optimistic tone.",
    "Positive": "The article highlights {entity}'s solid performance, with terms like {phrases}{contrast}, indicating a positive outlook.",
    "Neutral": "The article discusses {entity} with balanced language, including terms like {phrases}{contrast}, suggesting a neutral stance.",
    "Negative": "The article points to challenges for {entity}, with negative terms like {phrases}{contrast}, reflecting a bearish sentiment.",
    "Strongly Negative": "The article underscores significant issues for {entity}, with strong negative language like {phrases}{contrast}, indicating a highly bearish tone."
}

# New: Extract company/topic using spaCy NER
# Looks for organizations in the text (e.g., "Tesla") or defaults to "the company"
def extract_entity(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return entities[0] if entities else "the company"

# New: Extract key phrases using attention weights, with filtering
# Gets the most influential words/phrases, skips punctuation, and prefers multi-word phrases
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

# New: Detect contrasting sentiments in sentences
# Checks if some sentences are positive while others are negative (e.g., "Tesla thrives, industry struggles")
def detect_contrast(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return ""
    results = sentiment_pipeline(sentences)
    labels = [r['label'].capitalize() for r in results]
    if "Positive" in labels and "Negative" in labels:
        return ", contrasted with challenges in the broader industry"
    return ""

# Analyze sentiment and generate enhanced explanation
def analyze_sentiment(text):
    # Tokenize for attention weights
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_attention_mask=True)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1]  # Last layer attentions

    # Get sentiment label and score
    result = sentiment_pipeline(text)[0]
    label = result['label'].capitalize()
    score = result['score']

    # Map to final label
    if label == 'Positive' and score >= 0.75:
        final_label = "Strongly Positive"
    elif label == 'Positive':
        final_label = "Positive"
    elif label == 'Negative' and score >= 0.75:
        final_label = "Strongly Negative"
    elif label == 'Negative':
        final_label = "Negative"
    else:
        final_label = "Neutral"

    # Approximate pos, neu, neg for chart
    if label == 'Positive':
        pos = score
        neg = 0
        neu = 1 - score
    elif label == 'Negative':
        neg = score
        pos = 0
        neu = 1 - score
    else:
        neu = score
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
        'score': f"{score * 100:.2f}%",
        'explanation': explanation,
        'pos': pos,
        'neu': neu,
        'neg': neg
    }

# Flask route for web app
@app.route('/', methods=['GET', 'POST'])
def index():
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
        
        return render_template('index.html', results=results, chart_data=chart_data)
    
    return render_template('index.html', results=None, chart_data=None)

if __name__ == '__main__':
    app.run(debug=True)
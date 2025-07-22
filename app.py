from flask import Flask, render_template, request
from transformers import pipeline
import json
import re

app = Flask(__name__)
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label'].capitalize()
    score = result['score']
    
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
    
    # Approximate pos, neu, neg
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
    
    # Explanation with key words
    explanation = f"{final_label} with confidence {score:.2f}, based on financial sentiment analysis."
    
    positive_words = ['growth', 'profit', 'bull', 'surge', 'strong', 'beat', 'rise']
    negative_words = ['loss', 'decline', 'bear', 'crash', 'weak', 'miss', 'fall']
    
    found_pos = [word for word in positive_words if word in text.lower()]
    found_neg = [word for word in negative_words if word in text.lower()]
    
    if found_pos:
        explanation += f" Key positive indicators: {', '.join(found_pos)}."
    if found_neg:
        explanation += f" Key negative indicators: {', '.join(found_neg)}."
    
    return {
        'label': final_label,
        'score': f"{score * 100:.2f}%",
        'explanation': explanation,
        'pos': pos,
        'neu': neu,
        'neg': neg
    }

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
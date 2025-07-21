from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import re

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.75:
        label = "Strongly Positive"
    elif compound > 0.25:
        label = "Positive"
    elif compound <= -0.75:
        label = "Strongly Negative"
    elif compound < -0.25:
        label = "Negative"
    else:
        label = "Neutral"
    
    confidence = abs(compound) * 100  # Simple confidence as % of |compound|
    explanation = f"Based on positive: {scores['pos']:.2f}, neutral: {scores['neu']:.2f}, negative: {scores['neg']:.2f} scores."
    
    return {
        'label': label,
        'score': f"{confidence:.2f}%",
        'explanation': explanation,
        'pos': scores['pos'],
        'neu': scores['neu'],
        'neg': scores['neg']
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
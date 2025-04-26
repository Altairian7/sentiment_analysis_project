import re
import nltk
import dateparser
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, logging

# Suppress unimportant warnings from transformers
logging.set_verbosity_error()

# Download VADER lexicon if not already present
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize HuggingFace DistilBERT pipeline
print("⚙️ Loading DistilBERT sentiment model...")
try:
    distilbert_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"❌ Failed to load DistilBERT model: {e}")
    exit(1)

# Updated list of important keywords and phrases
important_keywords = [
    "urgent", "critical", "important", "must", "need", "required", "wish", "dream", "plan",
    "goal", "essential", "necessary", "vital", "priority", "mission", "intention", "objective",
    "ambition", "aspiration", "resolve", "dedication", "commitment", "focus", "vision",
    "strategy", "urgent need", "life goal", "meaningful", "significant", "pressing",
    "non-negotiable", "top priority"
]

# Detect human-readable event age from text using regex and dateparser
def get_event_age(text):
    # Find date-like phrases
    patterns = [
        r"\b(last year|this year|next year|last month|yesterday|today|last week|[0-9]{4})\b",
        r"\b(in \d{4})\b", r"\b(on \w+ \d{1,2}(st|nd|rd|th)?(, \d{4})?)\b"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(0)
            parsed_date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'past'})
            if parsed_date:
                today = datetime.now()
                diff = today - parsed_date
                days_ago = diff.days
                if days_ago == 0:
                    return "📅 Happened Today"
                elif days_ago == 1:
                    return "📅 Happened Yesterday"
                elif days_ago < 7:
                    return f"📅 Happened {days_ago} days ago"
                elif days_ago < 30:
                    return f"📅 Happened {days_ago // 7} week(s) ago"
                elif days_ago < 365:
                    return f"📅 Happened {days_ago // 30} month(s) ago"
                else:
                    return f"📅 Happened {days_ago // 365} year(s) ago"
    
    return "📅 No clear event date detected"

# Calculate importance score based on keyword presence
def get_importance_score(text):
    text_lower = text.lower()
    importance_count = 0
    for keyword in important_keywords:
        if keyword in text_lower:
            importance_count += 1

    normalized_score = min(importance_count / 5, 1.0)
    return round(normalized_score, 2)

# Label importance level from score
def get_importance_label(score):
    if score >= 0.8:
        return "🔥 Very Important"
    elif score >= 0.4:
        return "⭐ Important"
    else:
        return "✅ Not Important"

# Analyze full sentiment, memory, and importance
def analyze_sentiment(text):
    # VADER
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_sentiment = (
        "Positive" if vader_scores['compound'] >= 0.05 else
        "Negative" if vader_scores['compound'] <= -0.05 else
        "Neutral"
    )

    # DistilBERT
    distilbert_result = distilbert_analyzer(text)[0]
    distilbert_label = distilbert_result['label']
    distilbert_score = distilbert_result['score']

    # Importance & Time
    importance = get_importance_score(text)
    event_age = get_event_age(text)

    # Output
    print(f"\n📝 Input Sentence: {text}")
    print("🔹 VADER Sentiment:")
    print(f"   Compound Score: {vader_scores['compound']:.4f}")
    print(f"   Sentiment: {vader_sentiment}")
    print("🔹 DistilBERT Sentiment:")
    print(f"   Label: {distilbert_label}")
    print(f"   Confidence Score: {distilbert_score:.4f}")
    print("🔹 Importance Score:")
    print(f"   {importance}")
    print("🔹 Event Memory:")
    print(f"   {event_age}")
    print("🔹 Importance Label:")
    print(f"   {get_importance_label(importance)}")

# CLI entry
if __name__ == "__main__":
    print("\n💬 Sentiment Analysis Tool (VADER + DistilBERT + Memory + Importance)\n")
    while True:
        try:
            user_input = input("Enter a sentence (or type 'exit' to quit):\n> ")
            if user_input.strip().lower() == "exit":
                print("👋 Exiting...")
                break
            if user_input.strip() == "":
                print("⚠️ Please enter a non-empty sentence.")
                continue
            analyze_sentiment(user_input)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

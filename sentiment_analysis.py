import nltk
import dateparser
from dateparser.search import search_dates  # <-- add this
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, logging


# Suppress unimportant warnings from transformers
logging.set_verbosity_error()

# Download necessary resources
nltk.download('vader_lexicon')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize DistilBERT
print("⚙️ Loading DistilBERT sentiment model...")
try:
    distilbert_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"❌ Failed to load DistilBERT model: {e}")
    exit(1)

# Initialize NER pipeline for location extraction
print("⚙️ Loading NER model for location detection...")
try:
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )
except Exception as e:
    print(f"❌ Failed to load NER model: {e}")
    exit(1)

important_keywords = [
    "urgent", "critical", "important", "must", "need", "required", "wish", "dream", "plan",
    "goal", "essential", "necessary", "vital", "priority", "mission", "intention", "objective",
    "ambition", "aspiration", "resolve", "dedication", "commitment", "focus", "vision",
    "strategy", "urgent need", "life goal", "meaningful", "significant", "pressing",
    "non-negotiable", "top priority"
]

def get_event_age(text):
    # Use dateparser search_dates to find any dates
    results = dateparser.search.search_dates(text, settings={'PREFER_DATES_FROM': 'past'})
    if results:
        # Pick the first detected date
        parsed_date = results[0][1]
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
    else:
        return "📅 No clear event date detected"

def get_event_location(text):
    entities = ner_pipeline(text)
    locations = [ent['word'] for ent in entities if ent['entity_group'] == 'LOC']
    if locations:
        return "🌍 Location Detected: " + ", ".join(set(locations))
    else:
        return "🌍 No clear location detected"

def get_importance_score(text):
    text_lower = text.lower()
    importance_count = sum(1 for keyword in important_keywords if keyword in text_lower)
    normalized_score = min(importance_count / 5, 1.0)
    return round(normalized_score, 2)

def get_importance_label(score):
    if score >= 0.8:
        return "🔥 Very Important"
    elif score >= 0.4:
        return "⭐ Important"
    else:
        return "✅ Not Important"

def analyze_sentiment(text):
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_sentiment = (
        "Positive" if vader_scores['compound'] >= 0.05 else
        "Negative" if vader_scores['compound'] <= -0.05 else
        "Neutral"
    )

    distilbert_result = distilbert_analyzer(text)[0]
    distilbert_label = distilbert_result['label']
    distilbert_score = distilbert_result['score']

    importance = get_importance_score(text)
    event_age = get_event_age(text)
    event_location = get_event_location(text)

    print(f"\n📝 Input Sentence: {text}")
    print("🔹 VADER Sentiment:")
    print(f"   Compound Score: {vader_scores['compound']:.4f}")
    print(f"   Sentiment: {vader_sentiment}")
    print("🔹 DistilBERT Sentiment:")
    print(f"   Label: {distilbert_label}")
    print(f"   Confidence Score: {distilbert_score:.4f}")
    print("🔹 Importance Score:")
    print(f"   {importance}")
    print("🔹 Event Memory (Date Info):")
    print(f"   {event_age}")
    print("🔹 Event Location:")
    print(f"   {event_location}")
    print("🔹 Importance Label:")
    print(f"   {get_importance_label(importance)}")

if __name__ == "__main__":
    print("\n💬 Sentiment Analysis Tool (VADER + DistilBERT + Event Memory + Location + Importance)\n")
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

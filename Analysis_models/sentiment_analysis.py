import nltk
import dateparser
from dateparser.search import search_dates
from datetime import datetime
import os
import sqlite3
import json
import hashlib
import emoji
import re
from functools import lru_cache
from langdetect import detect, LangDetectException
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from transformers import pipeline, logging
from gensim.summarization import summarize as gensim_summarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Suppress unimportant warnings from transformers
logging.set_verbosity_error()

# Download necessary resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".sentiment_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize database
DB_PATH = os.path.join(CACHE_DIR, "sentiment_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        text TEXT,
        vader_compound REAL,
        vader_sentiment TEXT,
        distilbert_label TEXT,
        distilbert_score REAL,
        importance_score REAL,
        importance_label TEXT,
        event_age TEXT,
        event_location TEXT,
        language TEXT,
        topics TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()



# Cache model loading with function decorator
@lru_cache(maxsize=2)
def load_sentiment_model():
    print("⚙️ Loading DistilBERT sentiment model...")
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        print(f"❌ Failed to load DistilBERT model: {e}")
        return None

@lru_cache(maxsize=2)
def load_ner_model():
    print("⚙️ Loading NER model for location detection...")
    try:
        return pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
    except Exception as e:
        print(f"❌ Failed to load NER model: {e}")
        return None
    
    
    
    
    
# Load models with caching
distilbert_analyzer = load_sentiment_model()
ner_pipeline = load_ner_model()

# Enhanced list of important keywords with categorization
important_keywords = {
    "urgency": [
        "urgent", "critical", "immediately", "asap", "emergency", "pressing", 
        "deadline", "crucial", "vital", "priority"
    ],
    "necessity": [
        "must", "need", "required", "essential", "necessary", "mandatory",
        "non-negotiable", "imperative", "required", "obligatory"
    ],
    "aspiration": [
        "wish", "dream", "plan", "goal", "intention", "objective", "ambition",
        "aspiration", "resolve", "dedication", "commitment", "focus", "vision",
        "strategy", "purpose", "mission"
    ],
    "significance": [
        "important", "meaningful", "significant", "major", "substantial",
        "considerable", "notable", "remarkable", "momentous", "pivotal",
        "top priority", "life goal", "milestone"
    ]
}



all_important_keywords = [keyword for category in important_keywords.values() for keyword in category]

def get_keyword_category(word):
    """Return the category of an important keyword"""
    for category, keywords in important_keywords.items():
        if word in keywords:
            return category
    return None

def detect_language(text):
    """Detect the language of the input text with error handling"""
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails

def get_event_age(text):
    """Extract and calculate the age of events mentioned in the text"""
    try:
        # Use dateparser search_dates to find any dates
        results = dateparser.search.search_dates(text, settings={'PREFER_DATES_FROM': 'past'})
        if results:
            # Sort dates by earliest first
            sorted_dates = sorted(results, key=lambda x: x[1])
            parsed_date = sorted_dates[0][1]  # Use the earliest date
            today = datetime.now()
            diff = today - parsed_date
            days_ago = diff.days
            
            if days_ago < 0:
                return f"📅 Future event: {abs(days_ago)} days from now", parsed_date
            elif days_ago == 0:
                return "📅 Happened Today", parsed_date
            elif days_ago == 1:
                return "📅 Happened Yesterday", parsed_date
            elif days_ago < 7:
                return f"📅 Happened {days_ago} days ago", parsed_date
            elif days_ago < 30:
                return f"📅 Happened {days_ago // 7} week(s) ago", parsed_date
            elif days_ago < 365:
                return f"📅 Happened {days_ago // 30} month(s) ago", parsed_date
            else:
                return f"📅 Happened {days_ago // 365} year(s) ago", parsed_date
        else:
            return "📅 No clear event date detected", None
    except Exception as e:
        print(f"Date parsing error: {e}")
        return "📅 Error detecting event date", None
    
    
    
def get_event_location(text):
    """Extract locations from text with confidence scores"""
    if ner_pipeline is None:
        return "🌍 Location detection unavailable", []
    
    try:
        entities = ner_pipeline(text)
        locations = []
        
        # Group identical locations and calculate confidence
        location_confidence = {}
        for ent in entities:
            if ent['entity_group'] == 'LOC':
                loc = ent['word']
                score = ent['score']
                if loc in location_confidence:
                    # Average the confidence if we've seen this location before
                    location_confidence[loc] = (location_confidence[loc] + score) / 2
                else:
                    location_confidence[loc] = score
        
        # Convert to a list of (location, confidence) tuples and sort by confidence
        locations = [(loc, score) for loc, score in location_confidence.items()]
        locations.sort(key=lambda x: x[1], reverse=True)
        
        if locations:
            location_strs = [f"{loc} ({conf:.2f})" for loc, conf in locations]
            return "🌍 Locations detected: " + ", ".join(location_strs), locations
        else:
            return "🌍 No clear location detected", []
    except Exception as e:
        print(f"Location detection error: {e}")
        return "🌍 Error detecting locations", []

def get_importance_score(text):
    """Calculate importance score with stopword filtering"""
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = " ".join(filtered_words)
    
    # Count occurrences of important keywords
    keyword_matches = {}
    for category, keywords in important_keywords.items():
        category_count = sum(1 for keyword in keywords if keyword in filtered_text)
        if category_count > 0:
            keyword_matches[category] = category_count
    
    total_matches = sum(keyword_matches.values())
    
    # Calculate normalized score
    text_length = max(len(filtered_words), 1)  # Avoid division by zero
    normalized_score = min(total_matches / max(text_length / 10, 1), 1.0)
    
    return round(normalized_score, 2), keyword_matches



def get_importance_label(score):
    """Convert importance score to a descriptive label"""
    if score >= 0.8:
        return "🔥 Very Important"
    elif score >= 0.4:
        return "⭐ Important"
    elif score >= 0.2:
        return "📌 Somewhat Important"
    else:
        return "✅ Standard Priority"

def analyze_emoji_sentiment(text):
    """Analyze sentiment conveyed by emojis in the text"""
    emoji_list = emoji.emoji_list(text)
    
    if not emoji_list:
        return "No emojis detected", 0
    
    # Simple emoji sentiment dictionary (could be expanded)
    positive_emojis = {"😊", "😁", "😄", "😃", "😀", "🙂", "😍", "🥰", "❤️", "👍", "✅", "🎉", "🎊"}
    negative_emojis = {"😢", "😭", "😞", "😔", "😟", "😠", "😡", "👎", "❌", "💔", "😒", "😣"}
    
    emoji_count = len(emoji_list)
    positive_count = sum(1 for e in emoji_list if e['emoji'] in positive_emojis)
    negative_count = sum(1 for e in emoji_list if e['emoji'] in negative_emojis)
    
    # Calculate net sentiment
    if emoji_count == 0:
        emoji_sentiment = 0
    else:
        emoji_sentiment = (positive_count - negative_count) / emoji_count
    
    # Return label and score
    if emoji_sentiment > 0.3:
        return "Positive emoji sentiment", emoji_sentiment
    elif emoji_sentiment < -0.3:
        return "Negative emoji sentiment", emoji_sentiment
    else:
        return "Neutral emoji sentiment", emoji_sentiment
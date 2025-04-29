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
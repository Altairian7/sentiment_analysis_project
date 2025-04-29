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
# keyword_sentiment_tracker.py
import snscrape.modules.twitter as sntwitter
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

model = load('model/sentiment_model.pkl')

def keyword_tracker(keyword, limit=200):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(keyword).get_items():
        if len(tweets) == limit:
            break
        tweets.append(tweet.content)
    
    sentiments = [model.predict([t])[0] for t in tweets]
    df = pd.DataFrame({'Tweet': tweets, 'Sentiment': sentiments})
    df['Sentiment'].value_counts().plot(kind='bar')
    plt.title(f"Sentiment on '{keyword}'")
    plt.show()

if __name__ == "__main__":
    keyword = input("Enter keyword to track: ")
    keyword_tracker(keyword)

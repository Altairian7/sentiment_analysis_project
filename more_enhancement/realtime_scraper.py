# realtime_scraper.py
import snscrape.modules.twitter as sntwitter
import pandas as pd

def scrape_tweets(keyword, limit=100):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(keyword).get_items():
        if len(tweets) == limit:
            break
        tweets.append([tweet.date, tweet.content, tweet.user.username])
    
    df = pd.DataFrame(tweets, columns=['Date', 'Tweet', 'User'])
    return df

if __name__ == "__main__":
    keyword = input("Enter keyword to scrape: ")
    df = scrape_tweets(keyword, limit=50)
    print(df.head())
    df.to_csv('data/scraped_tweets.csv', index=False)

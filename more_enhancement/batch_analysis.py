# batch_analysis.py
import pandas as pd
from joblib import load

model = load('model/sentiment_model.pkl')

def batch_sentiment(file_path):
    df = pd.read_csv(file_path)
    df['Sentiment'] = df['Text'].apply(lambda x: model.predict([x])[0])
    output_path = file_path.replace(".csv", "_with_sentiments.csv")
    df.to_csv(output_path, index=False)
    print(f"Sentiment analysis done! File saved to {output_path}")

if __name__ == "__main__":
    file_path = input("Enter CSV file path: ")
    batch_sentiment(file_path)

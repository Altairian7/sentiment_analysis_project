# model_tuner.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

def model_comparison(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVC": SVC(),
        "Naive Bayes": MultinomialNB()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name}: {acc:.2f}")

if __name__ == "__main__":
    df = pd.read_csv('data/reviews.csv')
    X = df['Text']
    y = df['Sentiment']
    model_comparison(X, y)

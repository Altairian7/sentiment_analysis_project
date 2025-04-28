# explainability_tool.py
import lime
import lime.lime_text
import numpy as np
from joblib import load

model = load('model/sentiment_model.pkl')
vectorizer = load('model/vectorizer.pkl')

class_names = ['Negative', 'Neutral', 'Positive']

def predict_proba(texts):
    vectors = vectorizer.transform(texts)
    return model.predict_proba(vectors)

def explain_instance(text):
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=6)
    exp.show_in_notebook()

if __name__ == "__main__":
    text = input("Enter text to explain: ")
    explain_instance(text)

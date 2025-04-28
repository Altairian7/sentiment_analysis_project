# emotion_classifier.py
from transformers import pipeline

def predict_emotion(text):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    result = classifier(text)
    return result[0]['label']

if __name__ == "__main__":
    text = input("Enter text: ")
    emotion = predict_emotion(text)
    print(f"Detected emotion: {emotion}")

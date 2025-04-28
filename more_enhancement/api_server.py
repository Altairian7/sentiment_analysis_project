# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI()
model = load('model/sentiment_model.pkl')

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(input: TextInput):
    prediction = model.predict([input.text])[0]
    return {"sentiment": prediction}

# Run: uvicorn api_server:app --reload

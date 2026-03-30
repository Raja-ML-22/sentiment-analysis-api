from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import json


class TextInput(BaseModel):
    text: str


class SentimentNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


with open("artifacts/config.json") as f:
    config = json.load(f)

input_size = config["input_size"]

with open("artifacts/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

model = SentimentNet(input_size)
model.load_state_dict(torch.load("artifacts/sentiment_model.pth", map_location=torch.device("cpu")))
model.eval()

app = FastAPI(title="Sentiment Analysis API")


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


def predict(text: str):
    text = text.lower()
    X = vectorizer.transform([text]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor).squeeze()
        prob = torch.sigmoid(logits).item()

    sentiment = "positive" if prob > 0.5 else "negative"
    return sentiment, prob


@app.post("/predict")
def predict_sentiment(data: TextInput):
    sentiment, prob = predict(data.text)
    return {
        "sentiment": sentiment,
        "confidence": round(prob, 4)
    }
# Sentiment Analysis API

An end-to-end Sentiment Analysis system built using PyTorch, TF-IDF, FastAPI, and the IMDb Movie Reviews dataset.

The project trains a neural network classifier on movie reviews, saves the trained model and vectorizer, and exposes predictions through a public FastAPI API deployed on Render.

---

## Features

* Binary sentiment classification: Positive / Negative
* Trained on IMDb 50K movie reviews dataset
* TF-IDF text vectorization
* PyTorch neural network model
* FastAPI backend for inference
* Public deployment on Render
* Interactive Swagger API documentation

---

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 82.29% |
| Precision | 83.88% |
| Recall    | 79.94% |
| F1-Score  | 81.86% |

Training Details:

* Train samples: 40,000
* Test samples: 10,000
* TF-IDF Features: 5,000
* Framework: PyTorch

---

## Project Structure

```text
sentiment-analysis-api/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── artifacts/
│   ├── sentiment_model.pth
│   ├── vectorizer.pkl
│   └── config.json
│
├── data/
│   └── IMDB Dataset.csv
│
├── notebook/
│   └── Deep Learning.ipynb
│
└── src/
```

---

## Tech Stack

* Python
* PyTorch
* Scikit-learn
* FastAPI
* Uvicorn
* Pandas
* NumPy
* Render

---

## Live Demo

* API Home: [https://sentiment-analysis-api-cozh.onrender.com](https://sentiment-analysis-api-cozh.onrender.com)
* Swagger Docs: [https://sentiment-analysis-api-cozh.onrender.com/docs](https://sentiment-analysis-api-cozh.onrender.com/docs)

---

## API Endpoint

### POST `/predict`

Request:

```json
{
  "text": "This movie was fantastic and inspiring"
}
```

Example Response:

```json
{
  "sentiment": "positive",
  "confidence": 0.9824
}
```

Another Example:

```json
{
  "text": "Worst movie ever. Completely boring."
}
```

Response:

```json
{
  "sentiment": "negative",
  "confidence": 0.9441
}
```

---

## Local Installation

Clone the repository:

```bash
git clone https://github.com/Raja-ML-22/sentiment-analysis-api.git
cd sentiment-analysis-api
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Training Workflow

1. Load IMDb movie review dataset
2. Clean and preprocess review text
3. Convert text into TF-IDF vectors
4. Train a PyTorch neural network classifier
5. Evaluate performance on test data
6. Save trained model and vectorizer in `artifacts/`
7. Load artifacts in FastAPI for prediction
8. Deploy publicly using Render

---

## Deployment on Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## Future Improvements

* Add probability bar visualization in API response
* Build a Streamlit frontend
* Add batch prediction endpoint
* Use advanced embeddings such as BERT
* Dockerize the application
* Add CI/CD with GitHub Actions

---

## Resume Highlights

* Built and deployed an end-to-end sentiment analysis API using PyTorch, TF-IDF, FastAPI, and the IMDb 50K reviews dataset.
* Trained a neural network classifier on 40K reviews and evaluated on 10K test samples, achieving 82.29% accuracy and 81.86% F1-score.
* Deployed the project publicly on Render with interactive Swagger documentation and real-time sentiment prediction.

---

## Author

Esakki Raja P
BE AI & ML Student
Francis Xavier Engineering College
GitHub: [https://github.com/Raja-ML-22](https://github.com/Raja-ML-22)

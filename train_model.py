from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "phishing-detection" / "model"
DATASET_PATH = BASE_DIR / "emails.csv"


if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Missing training dataset: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

if not {"text", "label"}.issubset(df.columns):
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(df["text"], df["label"])

MODEL_DIR.mkdir(parents=True, exist_ok=True)
with (MODEL_DIR / "phishing_model.pkl").open("wb") as file_handle:
    pickle.dump(model, file_handle)

print("Model saved successfully!")

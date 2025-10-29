#!/usr/bin/env python3
"""Train a baseline logistic regression spam classifier and save model artifacts.

Usage:
    python train.py

This script downloads the dataset, preprocesses text with TF-IDF, trains LogisticRegression,
prints evaluation metrics, and saves model + vectorizer to disk.
"""
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json

# Config
DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)
OUT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = OUT_DIR / "model.joblib"
VECT_PATH = OUT_DIR / "vectorizer.joblib"
RANDOM_STATE = 42


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, header=None, names=["label", "message"])  # CSV without header
    df = df.dropna()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def train():
    print("Loading data from:", DATA_URL)
    df = load_data(DATA_URL)
    X = df["message"].astype(str).values
    y = df["label_num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_t = vectorizer.fit_transform(X_train)
    X_test_t = vectorizer.transform(X_test)

    print("Training LogisticRegression baseline...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_t, y_train)

    y_pred = clf.predict(X_test_t)
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1: {f:.4f}")

    # Save artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECT_PATH}")
    # Save metrics for Streamlit UI
    metrics = {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    METRICS_PATH = OUT_DIR / "metrics.json"
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    # Save a few example predictions from the test set for demos
    examples = []
    for text, true, pred in zip(X_test[:20], y_test[:20], y_pred[:20]):
        examples.append({"text": str(text), "true": int(true), "pred": int(pred)})
    EXAMPLES_PATH = OUT_DIR / "examples.json"
    with open(EXAMPLES_PATH, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(examples)} example predictions to {EXAMPLES_PATH}")


if __name__ == "__main__":
    train()

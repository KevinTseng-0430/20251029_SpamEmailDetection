Spam Classification — Phase 1 baseline

This directory contains a minimal, reproducible baseline for SMS/email spam classification using a scikit-learn Logistic Regression model.

What you get

- Training script (`train.py`) that downloads the dataset, preprocesses text with TF-IDF, trains a logistic regression model, evaluates it, and saves artifacts: `model.joblib`, `vectorizer.joblib`, `metrics.json`, and `examples.json`.
- Interactive demo (`streamlit_app.py`) which provides:
	- Class distribution histogram
	- Confusion matrix, ROC and Precision-Recall plots
	- Top tokens for spam/ham
	- Two example generators (generate a ham or spam message)
	- A colored confidence bar (red for spam, green for ham)
- `requirements.txt` listing required Python packages

Quick start (recommended)

1. Create a virtual environment and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r openspec/changes/add-spam-classification/requirements.txt
```

2. Train the baseline model (creates artifacts in this directory):

```bash
python3 openspec/changes/add-spam-classification/train.py
```

3. Run the Streamlit demo app:

```bash
streamlit run openspec/changes/add-spam-classification/streamlit_app.py
```

Files in this directory

- `train.py` — Training & artifact exporter (model.joblib, vectorizer.joblib, metrics.json, examples.json)
- `streamlit_app.py` — Interactive demo UI
- `requirements.txt` — Python dependencies
- `model.joblib`, `vectorizer.joblib`, `metrics.json`, `examples.json` — generated artifacts after training

Model results (from a baseline run)

- Accuracy: ~0.9776
- Precision: 1.0000
- Recall: ~0.8322
- F1: ~0.9084

Notes and next steps

- The training script currently recreates a train/test split at each run. For reproducible plots tied to the exact saved model, we can save the holdout test set and predictions during training; I can add that if you want.
- Consider adding hyperparameter tuning, text cleaning in the pipeline, and unit tests for preprocessing and inference.
- To containerize this app for deployment (Heroku/Streamlit Sharing/Docker), I can add a Dockerfile and Procfile.

License & attribution

This project uses the SMS Spam dataset from the PacktPublishing repository referenced in the proposal. Check dataset licensing before production use.

Contact

If you want further changes (styling, explainability, CI), tell me which item to prioritize.

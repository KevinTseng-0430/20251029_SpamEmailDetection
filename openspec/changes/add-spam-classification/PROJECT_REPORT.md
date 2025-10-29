Project Report — Spam Classification (Phase1-baseline)

Overview
--------
This project implements a Phase-1 baseline for SMS/email spam classification as part of the AIoT homework repository. The goal was to build a reproducible, minimal pipeline using classical ML (TF-IDF + Logistic Regression) and an interactive demo for exploration.

Dataset
-------
Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

The CSV contains two columns: `label` (ham/spam) and `message` (text). The training script downloads this CSV, cleans nulls, and maps `ham` to 0 and `spam` to 1.

Implementation
--------------
- `train.py` — Downloads the dataset, splits data (80/20 stratified), vectorizes text with TF-IDF (max_features=5000, stop words=english), trains a scikit-learn `LogisticRegression` (max_iter=1000), evaluates on a held-out test split, and writes artifacts:
  - `model.joblib` — trained classifier (joblib)
  - `vectorizer.joblib` — TF-IDF vectorizer (joblib)
  - `metrics.json` — evaluation metrics (accuracy, precision, recall, f1, n_train, n_test)
  - `examples.json` — example test messages + true/pred labels (for UI demos)

- `streamlit_app.py` — Interactive demo with:
  - Class distribution histogram (from dataset or fallback to examples)
  - Confusion matrix, ROC curve (AUC), precision-recall curve
  - Top tokens for spam and ham (by logistic regression coefficients)
  - Two example generator buttons (random ham or spam text inserted into input)
  - Prediction UI returning label + confidence and a colored confidence bar (red for spam/high-confidence, green for ham)

Reproducibility / How to run
----------------------------
1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r openspec/changes/add-spam-classification/requirements.txt
```

2. Train the baseline (this generates model artifacts):

```bash
python3 openspec/changes/add-spam-classification/train.py
```

3. Run the interactive demo:

```bash
streamlit run openspec/changes/add-spam-classification/streamlit_app.py
```

Key Results
-----------
A baseline run produced these metrics on the held-out test set (80/20 split):
- Accuracy: 0.9776
- Precision: 1.0000
- Recall: 0.8322
- F1: 0.9084

Model artifacts and metrics can be found in the change directory after training.

Limitations and risks
---------------------
- Baseline uses TF-IDF + Logistic Regression; modern improvements (transformers, embeddings) are not used.
- No production-level serving, authentication, or rate-limiting is implemented.
- Dataset is relatively small and drawn from SMS-style messages; email corpora differ in content and length.
- The live performance charts re-create a test split at runtime. For exact reproducibility, we should save the holdout test set and use it for plotting.

Next steps
----------
Suggested follow-up tasks (phase 2+):
1. Save holdout test set and predictions during training for reproducible charts.
2. Add text normalization (lowercasing, URL removal, punctuation handling) in a scikit-learn pipeline.
3. Hyperparameter tuning (GridSearchCV or randomized search) and cross-validation.
4. Add unit tests for preprocessing, training, and inference.
5. Containerize the app (Dockerfile) and add deployment instructions (Heroku/Streamlit Cloud/Docker Compose).
6. Add explainability: token-level SHAP or LIME for individual predictions.

Files changed/created in this change
-----------------------------------
- `openspec/changes/add-spam-classification/proposal.md` (proposal)
- `openspec/changes/add-spam-classification/tasks.md` (tasks)
- `openspec/changes/add-spam-classification/specs/spam/spec.md` (spec)
- `openspec/changes/add-spam-classification/train.py` (training script)
- `openspec/changes/add-spam-classification/streamlit_app.py` (interactive demo)
- `openspec/changes/add-spam-classification/requirements.txt` (dependencies)
- `openspec/changes/add-spam-classification/README.md` (updated)
- `openspec/changes/add-spam-classification/metrics.json` (generated)
- `openspec/changes/add-spam-classification/examples.json` (generated)

Contact / Maintainer
--------------------
For questions, ask in the repo or open an issue/PR. I can follow up with tests, containerization, or improved preprocessing on request.

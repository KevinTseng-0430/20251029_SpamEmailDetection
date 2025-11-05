# AIoT Homework Project

This repository contains exercises and change proposals for an AIoT homework project. It uses OpenSpec for spec-driven change management. The `openspec/` directory holds project conventions, specs, and change proposals.

Quick links
-----------
- OpenSpec instructions: `openspec/AGENTS.md`
- Project conventions: `openspec/project.md`
- Change proposals: `openspec/changes/`
- Spam classification change: `openspec/changes/add-spam-classification/`
- Device anomaly detection change: `openspec/changes/add-device-anomaly-detection/`

## Changes

### Spam classification
Change: `openspec/changes/add-spam-classification/`

This change adds a spam-classification capability (baseline) including training and an interactive Streamlit demo. See `openspec/changes/add-spam-classification/README.md` for quick start and details.

### Device anomaly detection
Change: `openspec/changes/add-device-anomaly-detection/`

This change adds a real-time anomaly detection system for device telemetry. See `openspec/changes/add-device-anomaly-detection/proposal.md` for more details.

Project demo (Streamlit)
-------------------------
Interactive demo : https://20251029spamemaildetection.streamlit.app/

CRISP‑DM Project Summary
------------------------
This project follows the CRISP‑DM process for a spam classification baseline (phase1).

1) Business Understanding
- Objective: Build a classifier that labels short messages (SMS/email) as `spam` or `ham` to reduce user exposure to unsolicited content and support downstream automation.
- Success criteria: A reproducible baseline with evaluation metrics (precision/recall/F1) and an interactive demo for exploration.

2) Data Understanding
- Dataset: SMS spam dataset (PacktPublishing example). CSV with two columns: `label` (ham/spam) and `message`.
- Observations: Class imbalance present (more ham than spam). Messages are short and noisy (abbreviations, punctuation, URLs).

3) Data Preparation
- Actions implemented in `train.py`:
	- Download CSV from the public URL
	- Drop nulls
	- Map labels to numeric (ham=0, spam=1)
	- TF‑IDF vectorization (max_features=5000, stop_words='english')
	- Stratified train/test split (80/20)

4) Modeling
- Baseline model: scikit‑learn LogisticRegression (max_iter=1000).
- Artifacts saved: `model.joblib`, `vectorizer.joblib`, plus `metrics.json` and `examples.json` for the UI.

5) Evaluation
- Metrics produced on the held‑out test split (example run):
	- Accuracy: 0.9776
	- Precision: 1.0000
	- Recall: 0.8322
	- F1: 0.9084
- The Streamlit demo includes a confusion matrix, ROC & PR curves, and top token tables for quick analysis.

6) Deployment
- The interactive demo is implemented with Streamlit and can be deployed to Streamlit Cloud. The app will auto-train on first run if model artifacts are missing (convenient for deployment but increases startup time).

How to run locally
------------------
1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r openspec/changes/add-spam-classification/requirements.txt
```

2. Train the baseline model:

```bash
python3 openspec/changes/add-spam-classification/train.py
```

3. Run the Streamlit demo:

```bash
streamlit run openspec/changes/add-spam-classification/streamlit_app.py
```

Notes & next steps
------------------
- Save the exact holdout test set and predictions during training to make dashboard charts reproducible across runs.
- Add text normalization and a scikit‑learn pipeline (cleaning, tokenizer, vectorizer) to reduce noise.
- Add hyperparameter tuning (GridSearchCV) and cross‑validation for a more robust baseline.
- Add unit tests and a simple CI workflow (GitHub Actions) to validate training and inference steps on push.
- Consider containerizing the app (Docker) and using Git LFS for large artifacts if you wish to track models in the repo.

Acknowledgements
----------------
Dataset: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

## Why
Spam messages degrade user experience and can pose security risks. Building an ML-driven spam classifier helps automate filtering, reduce manual triage, and improve downstream systems that rely on clean data.

## What Changes
- Add a new capability: spam email/SMS classification using machine learning.
- Phase 1 (baseline): build a logistic regression baseline classifier, train on the provided dataset, and produce evaluation metrics and a simple inference API.
- Later phases (placeholders): expand models, deploy to edge, add monitoring, and improve dataset/labeling.

**Dataset:**
- Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

**Note / Assumption:** The baseline for phase1 is a logistic regression classifier (per your request). If you'd like to evaluate SVM or other models later, we'll add those to subsequent phases.

## Phase structure
- phase1-baseline
  - Train a logistic regression classifier on the dataset linked above
  - Provide training script, evaluation, and a minimal inference script/notebook
- phase2
  - (placeholder)
- phase3
  - (placeholder)

## Impact
- Affected specs: `monitoring` (if we add model monitoring), `spam` (new capability)
- Affected code: ML training scripts, preprocessing utilities, minimal inference endpoint
- External dependencies: scikit-learn, pandas, numpy

## Acceptance Criteria
- A reproducible training script that downloads the dataset, preprocesses text, trains a logistic regression model, and outputs precision/recall/F1 on a held-out test set.
- An inference script/notebook demonstrating predictions on sample messages.
- Tasks documented in `tasks.md` and spec delta added under `openspec/changes/add-spam-classification/specs/spam/spec.md`.
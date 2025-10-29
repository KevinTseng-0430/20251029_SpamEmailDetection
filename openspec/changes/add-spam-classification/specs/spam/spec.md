## ADDED Requirements

### Requirement: Spam Classification Capability
The system SHALL classify incoming SMS/email messages as `spam` or `ham` using a machine learning model. The initial baseline model SHALL be a logistic regression classifier.

#### Scenario: Train baseline model
- **WHEN** the training script is run on the provided dataset
- **THEN** the pipeline downloads and preprocesses the data
- **AND** trains a logistic regression model and saves the model artifact
- **AND** outputs evaluation metrics including precision, recall, and F1-score on a held-out test split

#### Scenario: Inference with baseline model
- **WHEN** a user submits a message to the inference script
- **THEN** the system returns a prediction of `spam` or `ham` with a confidence score

#### Scenario: Evaluation meets minimum quality
- **WHEN** the trained baseline is evaluated
- **THEN** the model produces a reasonable baseline (no strict threshold set yet; recorded metrics in experiment log)

## MODIFIED Requirements

(none)

## REMOVED Requirements

(none)
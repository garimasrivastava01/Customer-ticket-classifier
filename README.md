# Customer Support Ticket Classifier

A complete ML pipeline to classify support tickets and detect urgency.

## Features
- **4 Categories**: billing, technical, complaint, general
- **4 Urgency Levels**: critical, high, medium, low
- **Models**: TF-IDF + Logistic Regression (Scikit-learn)
- **Preprocessing**: Regex cleaning (NLTK-compatible)
- **Recommended Actions** + SLA based on category × urgency matrix

## Files
- `train_model.py` — Trains and saves both models
- `classifier.py` — Core classifier API (`classify(text) → dict`)
- `category_model.pkl` — Saved category model
- `urgency_model.pkl` — Saved urgency model
- `model_metrics.json` — Accuracy + F1 scores

## Setup
```bash
pip install scikit-learn pandas numpy nltk
python train_model.py   # Train models (one time)
python classifier.py    # Test with sample tickets
```

## Usage
```python
from classifier import classify

result = classify("I was charged twice and my account is frozen!")
print(result['category']['label'])     # billing
print(result['urgency']['label'])      # high
print(result['recommended_action'])    # Route to billing specialist within 4 hours.
```

## Dataset
Training data is synthetic, built to mirror patterns from the
Kaggle "Customer Support on Twitter" dataset.
For production, replace with real labeled data from the Kaggle dataset.

## Model Performance
- Category Accuracy: ~70% (on synthetic test set)
- Urgency Accuracy: ~58% (on synthetic test set)
Performance improves significantly with real labeled data.

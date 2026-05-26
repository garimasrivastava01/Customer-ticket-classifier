"""
Core Classifier API - used by the Flask server
"""
import pickle
import re
import json
from pathlib import Path

BASE = Path(__file__).parent

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load models once
with open(BASE / 'category_model.pkl', 'rb') as f:
    CAT_MODEL = pickle.load(f)

with open(BASE / 'urgency_model.pkl', 'rb') as f:
    URG_MODEL = pickle.load(f)

CATEGORY_LABELS = {
    "billing": {"icon": "💳", "color": "#f59e0b", "desc": "Payment, invoices, refunds, charges"},
    "technical": {"icon": "⚙️", "color": "#3b82f6", "desc": "Bugs, errors, crashes, features"},
    "complaint": {"icon": "😤", "color": "#ef4444", "desc": "Dissatisfaction, negative experience"},
    "general": {"icon": "💬", "color": "#8b5cf6", "desc": "Questions, info, how-to queries"},
}

URGENCY_LABELS = {
    "critical": {"icon": "🔴", "color": "#dc2626", "sla": "< 1 hour", "priority": 4},
    "high":     {"icon": "🟠", "color": "#ea580c", "sla": "< 4 hours", "priority": 3},
    "medium":   {"icon": "🟡", "color": "#ca8a04", "sla": "< 24 hours", "priority": 2},
    "low":      {"icon": "🟢", "color": "#16a34a", "sla": "< 72 hours", "priority": 1},
}

def classify(text: str) -> dict:
    cleaned = clean_text(text)

    # Category
    cat_probs = CAT_MODEL.predict_proba([cleaned])[0]
    cat_classes = CAT_MODEL.classes_
    cat_label = cat_classes[cat_probs.argmax()]
    cat_confidence = float(cat_probs.max())

    # Urgency
    urg_probs = URG_MODEL.predict_proba([cleaned])[0]
    urg_classes = URG_MODEL.classes_
    urg_label = urg_classes[urg_probs.argmax()]
    urg_confidence = float(urg_probs.max())

    # All category probabilities
    cat_breakdown = {
        cls: round(float(prob) * 100, 1)
        for cls, prob in zip(cat_classes, cat_probs)
    }

    urg_breakdown = {
        cls: round(float(prob) * 100, 1)
        for cls, prob in zip(urg_classes, urg_probs)
    }

    return {
        "input": text,
        "category": {
            "label": cat_label,
            "confidence": round(cat_confidence * 100, 1),
            "breakdown": cat_breakdown,
            **CATEGORY_LABELS[cat_label]
        },
        "urgency": {
            "label": urg_label,
            "confidence": round(urg_confidence * 100, 1),
            "breakdown": urg_breakdown,
            **URGENCY_LABELS[urg_label]
        },
        "recommended_action": get_action(cat_label, urg_label)
    }

def get_action(category, urgency):
    actions = {
        ("billing", "critical"):   "Escalate to billing team IMMEDIATELY. Freeze account changes.",
        ("billing", "high"):       "Route to billing specialist within 4 hours.",
        ("billing", "medium"):     "Assign to billing queue. Resolve within 24h.",
        ("billing", "low"):        "Add to standard billing queue.",
        ("technical", "critical"): "Page on-call engineer NOW. Check status page.",
        ("technical", "high"):     "Assign senior engineer. Monitor closely.",
        ("technical", "medium"):   "Add to tech support queue with standard SLA.",
        ("technical", "low"):      "Log as feedback. Route to tech team.",
        ("complaint", "critical"): "Escalate to Customer Success Manager immediately.",
        ("complaint", "high"):     "Assign senior support agent. Personal follow-up required.",
        ("complaint", "medium"):   "Route to support team with empathy priority.",
        ("complaint", "low"):      "Acknowledge and add to complaint tracking.",
        ("general", "critical"):   "Respond immediately with clear information.",
        ("general", "high"):       "Respond within 4 hours.",
        ("general", "medium"):     "Route to knowledge base or FAQ team.",
        ("general", "low"):        "Provide self-service link or FAQ response.",
    }
    return actions.get((category, urgency), "Route to general support queue.")

if __name__ == '__main__':
    # Quick test
    tests = [
        "I was charged twice and need an urgent refund!",
        "App keeps crashing every time I open it",
        "This is absolutely unacceptable, I'm cancelling",
        "How do I reset my password?",
        "CRITICAL: Production database is down, entire business halted RIGHT NOW"
    ]
    for t in tests:
        r = classify(t)
        print(f"\nInput: {t}")
        print(f"  Category: {r['category']['label']} ({r['category']['confidence']}%)")
        print(f"  Urgency:  {r['urgency']['label']} ({r['urgency']['confidence']}%)")
        print(f"  Action:   {r['recommended_action']}")

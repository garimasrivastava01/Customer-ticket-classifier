"""
Customer Support Ticket Classifier
- Classifies tickets: billing, technical, complaint, general
- Detects urgency: low, medium, high, critical
- Uses TF-IDF + Logistic Regression (Scikit-learn)
- No external data needed — uses synthetic training data inspired by Twitter support corpus
"""

import pickle
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

# ─── Training Data (inspired by Customer Support on Twitter dataset patterns) ──

TRAINING_DATA = {
    "category": [
        # BILLING (60 samples)
        "I was charged twice for my subscription this month!",
        "My invoice shows wrong amount, I need a refund",
        "Why was my credit card billed without authorization?",
        "I cancelled my plan but still got charged",
        "Payment failed but money was deducted",
        "I need a receipt for my last payment",
        "Requesting refund for duplicate charge",
        "My promo code wasn't applied to the bill",
        "Subscription renewed even after cancellation",
        "Wrong tax amount on my invoice",
        "I want to upgrade my plan but can't find billing page",
        "Overcharged by $50 this month",
        "Auto-renewal turned on without my consent",
        "Refund not processed after 10 days",
        "Credit not showing on my account",
        "Bank says charge is pending but account not updated",
        "Invoice missing from my account history",
        "How do I change my payment method?",
        "Can I get annual billing instead of monthly?",
        "My free trial ended and I was billed automatically",
        "Charged for a service I never used",
        "Payment declined but no reason given",
        "I need GST invoice for my company",
        "The billing address on invoice is wrong",
        "When will my refund be credited back?",
        "I disputed a charge with my bank, please help",
        "Upgrade failed but I was charged",
        "My account shows negative balance",
        "Why is there a $1 charge on my card?",
        "Billing cycle changed without notice",
        "I need to add a secondary payment method",
        "My corporate card is being declined",
        "Coupon code says expired but it's valid",
        "I was charged during service outage",
        "Is there a student discount available?",
        "My wallet balance is incorrect",
        "Transaction history not loading",
        "Payment gateway showing error",
        "I want to switch from PayPal to card",
        "Will I be charged for pausing my account?",
        "Unexpected charges appeared on my bill",
        "I requested cancellation 30 days ago and still billed",
        "My plan was downgraded but I paid for premium",
        "Partial refund received, rest still pending",
        "Tax exemption not applied to my account",
        "International transaction fee charged unfairly",
        "My cashback is not reflecting",
        "The payment link in email is broken",
        "I need to update my VAT number",
        "Currency conversion rate seems wrong",
        "Subscription price changed without email",
        "I'm being charged for seats I removed",
        "Billing portal is inaccessible",
        "My enterprise invoice is wrong",
        "Please send me all invoices from last year",
        "Why does my bill increase every 3 months?",
        "Gift card balance not applied",
        "I want a pro-rated refund for unused days",
        "Account shows payment due but I already paid",
        "Charged for premium but getting free features only",

        # TECHNICAL (60 samples)
        "App keeps crashing whenever I open it",
        "Login not working, password reset email not arriving",
        "Website showing 500 error page",
        "API returning null response for all requests",
        "Can't upload files, getting timeout error",
        "Dark mode not saving my preference",
        "Two-factor authentication code not working",
        "Integration with Slack is broken",
        "Data export feature is stuck at 0%",
        "Profile picture won't upload",
        "Search results are completely wrong",
        "Notifications not being sent",
        "Mobile app shows blank screen on launch",
        "My dashboard data is outdated by 3 days",
        "CSV import fails with no error message",
        "Webhook is not triggering",
        "OAuth login returns error 401",
        "Video calls dropping every 2 minutes",
        "Browser extension stopped working after update",
        "I can't delete my account",
        "Reports are generating wrong figures",
        "The map feature shows wrong location",
        "Email templates not rendering correctly",
        "Bulk action button is greyed out",
        "My integrations page won't load",
        "API rate limit hit too fast",
        "Real-time sync not working between devices",
        "Data backup failed overnight",
        "Calendar sync with Google is broken",
        "Sorting doesn't work on the table view",
        "Comments section is not loading",
        "Code editor has broken syntax highlighting",
        "Team members can't see my shared files",
        "Form submission gives error 422",
        "SAML SSO login broken for my org",
        "Webhook payload is malformed",
        "QR code scanner not working in app",
        "Drag and drop is not functioning",
        "Charts show no data despite records existing",
        "Copy-paste not working in text editor",
        "My automations stopped running",
        "Custom domain not pointing correctly",
        "SSL certificate showing as expired",
        "Database connection timeout",
        "Image compression making images unreadable",
        "Chat history not loading",
        "Screen recording feature frozen",
        "My saved filters reset every session",
        "App is extremely slow on mobile data",
        "Tags getting removed automatically",
        "Forgot password link takes me to 404",
        "Print feature showing blank page",
        "Time zone settings not applying",
        "Keyboard shortcut stopped working",
        "Audio not playing in web app",
        "My data is missing after migration",
        "Cannot create more than 5 projects",
        "Table pagination broken at page 3",
        "Email verification link expired instantly",
        "Session logout happening randomly",

        # COMPLAINT (60 samples)
        "This is absolutely unacceptable service quality",
        "I've been waiting 5 days for a response, this is ridiculous",
        "Your support team is completely useless",
        "I want to cancel because your product is terrible",
        "This is the worst customer experience I've ever had",
        "Your company doesn't care about customers at all",
        "I've lost business because of your failure",
        "Stop sending me spam emails after I unsubscribed",
        "Your agent was rude and unhelpful",
        "I was promised a callback that never happened",
        "This product is completely misleading in ads",
        "I'm going to report your company to consumer protection",
        "Three tickets and no resolution, I'm disgusted",
        "Your chatbot is a waste of my time",
        "I explicitly said don't call me, you called anyway",
        "Your service is degrading every month",
        "Nobody responds to my urgent issues",
        "I was misled about pricing",
        "Your terms of service changed without notice",
        "I demand compensation for lost work",
        "Your product destroyed my important data",
        "I'm posting a review about this terrible experience",
        "This is fraud, I'm contacting my lawyer",
        "Your app violated my privacy",
        "Repeated apologies without any fix is not acceptable",
        "I feel like my complaints are being ignored",
        "The product demo was completely different from reality",
        "You are wasting my time with scripted responses",
        "This is my 10th follow-up with no resolution",
        "Your team contradicted each other on same issue",
        "Nobody in your support team knows your own product",
        "You escalated and then nothing happened",
        "I'm extremely frustrated with the lack of transparency",
        "This is not what was in the contract",
        "Your SLA guarantee means nothing apparently",
        "I was told 24 hours but it's been a week",
        "You're holding my data hostage during cancellation",
        "The new update ruined everything that worked",
        "I feel disrespected by your support staff",
        "This bug was reported months ago and still not fixed",
        "Your company's attitude toward SMBs is disgraceful",
        "Stop making me repeat myself every ticket",
        "Compensation offered is insulting for damages caused",
        "Your social media team is more helpful than support",
        "My account was suspended without any reason",
        "You are deleting my reviews on your page",
        "Your refund process is designed to frustrate users",
        "I can't believe this is how you treat loyal customers",
        "Everything was fine until you changed your team",
        "I was ghosted after submitting a complaint formally",
        "Escalation path is completely broken in your org",
        "I'll never recommend your product to anyone",
        "Your NPS survey is useless if you ignore feedback",
        "You owe me a real apology not a template",
        "Five stars on ads, one star in reality",
        "I regret signing the annual contract",
        "My entire team is switching to your competitor",
        "You're ignoring accessibility needs of disabled users",
        "Your outage was hidden from status page on purpose",
        "I want to speak to a human, not a bot",

        # GENERAL (40 samples)
        "How do I reset my password?",
        "What are your support hours?",
        "Can I use the service in India?",
        "Is there a free trial available?",
        "How do I add team members?",
        "What file formats do you support?",
        "Can I export my data at any time?",
        "Is there a mobile app available?",
        "How does pricing work for teams?",
        "What's the difference between plans?",
        "Do you offer phone support?",
        "Can I integrate with Zapier?",
        "Is my data encrypted?",
        "How do I delete my account?",
        "What happens after my trial ends?",
        "How many users can be on one account?",
        "Do you have a desktop application?",
        "Can I use offline mode?",
        "What countries do you operate in?",
        "Is there a student or nonprofit discount?",
        "How do I change my email address?",
        "Where can I find tutorials?",
        "What is your uptime guarantee?",
        "Can I white-label the product?",
        "How do I contact sales?",
        "Can I schedule a demo?",
        "Is there an API available?",
        "What languages does your app support?",
        "Do you have a community forum?",
        "How do I enable dark mode?",
        "Can I get a custom plan?",
        "Where are your servers located?",
        "Do you offer 24/7 support?",
        "How do I invite collaborators?",
        "What browsers are supported?",
        "Is there a Chrome extension?",
        "How do I report a bug?",
        "Do you have a roadmap I can view?",
        "Can I use keyboard shortcuts?",
        "How do I change notification settings?",
    ],
    "label": (
        ["billing"] * 60 +
        ["technical"] * 60 +
        ["complaint"] * 60 +
        ["general"] * 40
    )
}

URGENCY_DATA = {
    "text": [
        # CRITICAL
        "URGENT my entire database was wiped out right now!",
        "Emergency! Production server is completely down",
        "CRITICAL: All customer data is showing to wrong users",
        "ASAP help, we are losing $10,000 per hour due to outage",
        "URGENT: Security breach detected, accounts being accessed",
        "Our payment system is down, all transactions failing NOW",
        "EMERGENCY medical device software crashed, lives at risk",
        "Immediate help needed, entire team locked out of system",
        "Critical data loss happening right now, please respond immediately",
        "SOS! All our files got deleted, need urgent recovery",
        "Urgent system outage affecting 10,000 users right now",
        "Production database down, business completely halted",
        "CRITICAL BUG: users are seeing each other's private data",
        "Complete service failure, every minute costs us revenue",
        "Emergency: login page down before major product launch today",

        # HIGH
        "I've been waiting 3 days and issue is still unresolved",
        "This bug is blocking my entire team's work",
        "Charging issue affecting multiple accounts",
        "My client demo is tomorrow and the feature is broken",
        "We have a deadline in 2 hours and export is failing",
        "Issue not resolved after 5 days, very frustrated",
        "Multiple users reporting same login failure",
        "Service slow for last 2 days, very impacting",
        "I'm losing customers because of this bug",
        "Refund still not received after 2 weeks",
        "Cannot access my account for 24 hours",
        "Business operations blocked by this error",
        "Email delivery failing for all users since yesterday",
        "API down causing our product to fail for customers",
        "This is urgent, my license expires today",

        # MEDIUM
        "Feature not working as expected",
        "I was charged a wrong amount, need correction",
        "App crashes sometimes on my phone",
        "Notification emails going to spam",
        "PDF export has formatting issues",
        "I noticed a billing discrepancy last week",
        "Some data is not syncing correctly",
        "Mobile app is a bit slow",
        "Calendar integration has a minor issue",
        "Report generation takes too long",
        "Some images are not loading on the dashboard",
        "Profile settings not saving sometimes",
        "Search results are slightly off",
        "A few users can't login intermittently",
        "Minor UI bug on the settings page",

        # LOW
        "How do I change my password?",
        "Can I get an invoice for last month?",
        "Just curious about your pricing plans",
        "Would be nice to have dark mode",
        "Small typo on your pricing page",
        "General question about your API",
        "Just wanted to share feedback",
        "How do I update notification preferences?",
        "Where can I find documentation?",
        "Looking for tutorial videos",
        "Suggestion: add keyboard shortcuts",
        "I'd like to know about enterprise pricing",
        "What is your refund policy?",
        "Is there a way to customize the UI?",
        "How do I add a team member?",
    ],
    "label": (
        ["critical"] * 15 +
        ["high"] * 15 +
        ["medium"] * 15 +
        ["low"] * 15
    )
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_category_model():
    df = pd.DataFrame(TRAINING_DATA)
    df['clean'] = df['category'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1
        )),
        ('clf', LogisticRegression(
            C=5.0,
            max_iter=500,
            random_state=42,
            solver='lbfgs'
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("=== CATEGORY MODEL ===")
    print(classification_report(y_test, y_pred))

    return pipeline, report


def train_urgency_model():
    df = pd.DataFrame(URGENCY_DATA)
    df['clean'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=3000,
            sublinear_tf=True,
            min_df=1
        )),
        ('clf', LogisticRegression(
            C=3.0,
            max_iter=500,
            random_state=42,
            solver='lbfgs'
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("=== URGENCY MODEL ===")
    print(classification_report(y_test, y_pred))

    return pipeline, report


def save_models():
    cat_model, cat_report = train_category_model()
    urg_model, urg_report = train_urgency_model()

    with open('/home/claude/ticket_classifier/category_model.pkl', 'wb') as f:
        pickle.dump(cat_model, f)

    with open('/home/claude/ticket_classifier/urgency_model.pkl', 'wb') as f:
        pickle.dump(urg_model, f)

    metrics = {
        "category_accuracy": cat_report["accuracy"],
        "urgency_accuracy": urg_report["accuracy"],
        "category_f1_macro": cat_report["macro avg"]["f1-score"],
        "urgency_f1_macro": urg_report["macro avg"]["f1-score"],
    }

    with open('/home/claude/ticket_classifier/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n✓ Models saved successfully!")
    print(f"  Category Accuracy: {metrics['category_accuracy']:.2%}")
    print(f"  Urgency Accuracy:  {metrics['urgency_accuracy']:.2%}")
    return metrics


if __name__ == '__main__':
    save_models()

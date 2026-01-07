# dump_model_object.py
# ==========================================
# Dump English Sentiment Analysis Model
# ==========================================

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_and_dump():
    # ================= DATA =================
    texts = [
        "This product is very good",
        "Excellent quality and fast delivery",
        "Amazing experience, I love it",
        "Bad product, very disappointed",
        "Terrible quality, waste of money",
        "It is okay, not bad",
        "Average quality"
    ]

    labels = [
        "positive", "positive", "positive",
        "negative", "negative",
        "neutral", "neutral"
    ]

    # ================= TRAIN =================
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    # ================= PATH =================
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model_en.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer_en.pkl")

    # ================= DUMP =================
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("✅ Dump thành công!")
    print(f"📦 {model_path}")
    print(f"📦 {vectorizer_path}")


if __name__ == "__main__":
    train_and_dump()

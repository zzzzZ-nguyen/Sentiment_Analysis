# ==========================================
# Dump English Sentiment Analysis Model
# Train from Excel
# ==========================================

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_and_dump():
    # ================= LOAD EXCEL =================
    excel_path = "data/sentiment_train.xlsx"
    df = pd.read_excel(excel_path)

    # Validate columns
    if not {"text", "sentiment"}.issubset(df.columns):
        raise ValueError("Excel must contain columns: text, sentiment")

    texts = df["text"].astype(str).tolist()
    labels = df["sentiment"].astype(str).tolist()

    # ================= TRAIN =================
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000
    )
    model.fit(X, labels)

    # ================= SAVE =================
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model_en.pkl")
    joblib.dump(vectorizer, "models/vectorizer_en.pkl")

    print("✅ Train từ Excel & dump model thành công!")
    print(f"📊 Tổng số mẫu train: {len(texts)}")


if __name__ == "__main__":
    train_and_dump()

import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from docx import Document


# ==================================================
# 🔍 Language detection
# ==================================================
def is_vietnamese(text: str) -> bool:
    return bool(re.search(
        r"[àáạảãâăđêôơưíìịỉĩúùụủũýỳỵỷỹ]",
        text.lower()
    ))


# ==================================================
# 🇻🇳 Vietnamese sentiment (rule-based)
# ==================================================
VI_POS = [
    "tốt", "tuyệt", "hài lòng", "xuất sắc",
    "ổn", "đẹp", "ngon", "ưng ý", "hoàn hảo"
]

VI_NEG = [
    "tệ", "xấu", "kém", "thất vọng",
    "dở", "lỗi", "tồi", "không tốt"
]

def vietnamese_sentiment(text: str):
    score = 0
    t = text.lower()

    for w in VI_POS:
        if w in t:
            score += 1
    for w in VI_NEG:
        if w in t:
            score -= 1

    if score > 0:
        return "positive", min(0.6 + score * 0.1, 0.95)
    elif score < 0:
        return "negative", min(0.6 + abs(score) * 0.1, 0.95)
    else:
        return "neutral", 0.55


# ==================================================
# 🇺🇸 English ML model
# ==================================================
@st.cache_resource
def load_english_model():
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

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer


# ==================================================
# 🎯 PAGE
# ==================================================
def show():

    st.markdown(
        "<h3 style='color:#2b6f3e;'>Analysis – Sentiment Analysis (Vietnamese & English)</h3>",
        unsafe_allow_html=True
    )

    st.write("Analyze product reviews and classify sentiment.")

    model_en, vectorizer_en = load_english_model()

    # ================= INPUT SINGLE REVIEW =================
    st.subheader("📝 Input Product Review")

    review = st.text_area(
        "Enter a product review (Vietnamese or English):",
        height=120,
        placeholder="Ví dụ: Sản phẩm tốt / This product is excellent"
    )

    if st.button("▶️ Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            if is_vietnamese(review):
                sentiment, confidence = vietnamese_sentiment(review)
                st.success(f"Predicted Sentiment: **{sentiment.upper()}**")
                st.info(f"Confidence: **{confidence:.2f}**")
                st.caption("Language detected: Vietnamese")
            else:
                X = vectorizer_en.transform([review])
                sentiment = model_en.predict(X)[0]
                confidence = model_en.predict_proba(X).max()
                st.success(f"Predicted Sentiment: **{sentiment.upper()}**")
                st.info(f"Confidence: **{confidence:.2f}**")
                st.caption("Language detected: English")

    # ================= BATCH ANALYSIS =================
    st.write("---")
    st.subheader("📂 Upload Reviews Dataset (CSV / TXT / DOCX)")

    st.caption(
        "CSV: cột 'review' | TXT: mỗi dòng | DOCX: mỗi đoạn"
    )

    file = st.file_uploader(
        "Upload file",
        type=["csv", "txt", "docx"]
    )

    if file:
        reviews = []

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            if "review" not in df.columns:
                st.error("CSV must contain column named: review")
                return
            reviews = df["review"].astype(str).tolist()

        elif file.name.endswith(".txt"):
            content = file.read().decode("utf-8")
            reviews = [l.strip() for l in content.splitlines() if l.strip()]

        elif file.name.endswith(".docx"):
            doc = Document(file)
            reviews = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        sentiments, confidences = [], []

        for r in reviews:
            if is_vietnamese(r):
                s, c = vietnamese_sentiment(r)
            else:
                X = vectorizer_en.transform([r])
                s = model_en.predict(X)[0]
                c = model_en.predict_proba(X).max()

            sentiments.append(s)
            confidences.append(round(c, 2))

        result_df = pd.DataFrame({
            "review": reviews,
            "sentiment": sentiments,
            "confidence": confidences
        })

        st.success(f"Processed {len(result_df)} reviews")
        st.dataframe(result_df.head())

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(result_df["sentiment"].value_counts())

        st.download_button(
            "⬇️ Download result (CSV)",
            data=result_df.to_csv(index=False),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )

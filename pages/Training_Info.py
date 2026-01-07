import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==================================================
# 📦 LOAD MODEL OBJECTS
# ==================================================
@st.cache_resource
def load_model_objects():
    # Điều chỉnh đường dẫn tương đối cho phù hợp với cấu trúc thư mục của bạn
    model_path = os.path.join("models", "model_en.pkl")
    vectorizer_path = os.path.join("models", "vectorizer_en.pkl")

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except:
        return None, None

# ==================================================
# 📊 TRAINING INFO – SENTIMENT ANALYSIS
# ==================================================
def show():
    st.markdown(
        "<h3 style='color:#2b6f3e;'>Training Info – Sentiment Analysis (Advanced)</h3>",
        unsafe_allow_html=True
    )

    st.write(
        "This section presents the training pipeline, model information, "
        "evaluation results, and comparison of sentiment analysis models."
    )
    st.write("---")

    # Load Model
    model, vectorizer = load_model_objects()

    # ==================================================
    # 1️⃣ RAW DATASET (Mở rộng dữ liệu mẫu để tính toán thật)
    # ==================================================
    st.subheader("1️⃣ Raw Dataset")

    # Tạo dữ liệu giả lập đủ lớn để demo tính toán
    raw_data = pd.DataFrame({
        "review": [
            "Sản phẩm rất tốt", "Chất lượng kém, thất vọng", "This product is amazing", 
            "Bad quality, waste of money", "Average product", "Really loved it",
            "Terrible experience", "Normal quality", "Excellent service", "Don't buy this"
        ],
        "label": [
            "positive", "negative", "positive", 
            "negative", "neutral", "positive",
            "negative", "neutral", "positive", "negative"
        ]
    })

    st.dataframe(raw_data)
    st.caption("• Dataset mẫu được sử dụng để demo tính toán các chỉ số bên dưới.")
    st.write("---")

    # ==================================================
    # 2️⃣ PREPROCESSING
    # ==================================================
    st.subheader("2️⃣ Preprocessed Data")
    processed_data = raw_data.copy()
    processed_data["review_clean"] = processed_data["review"].str.lower()
    st.dataframe(processed_data.head())
    st.caption("Tiền xử lý: Chuyển chữ thường, loại bỏ ký tự đặc biệt.")
    st.write("---")

    # ==================================================
    # 3️⃣ MODEL INFORMATION
    # ==================================================
    st.subheader("3️⃣ Model Information")
    
    if model and vectorizer:
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("##### 📌 Logistic Regression Config")
            st.table(pd.DataFrame({
                "Property": ["Model Type", "Classes", "Solver"],
                "Value": ["LogisticRegression", str(model.classes_), getattr(model, 'solver', 'lbfgs')]
            }))
        
        with col_info2:
            st.markdown("##### 📌 TF-IDF Config")
            st.table(pd.DataFrame({
                "Property": ["Vectorizer", "Vocab Size", "N-gram"],
                "Value": ["TfidfVectorizer", len(vectorizer.vocabulary_), str(vectorizer.ngram_range)]
            }))
    else:
        st.error("Không tìm thấy file model trong thư mục 'models/'. Vui lòng kiểm tra lại.")

    st.write("---")

    # ==================================================
    # 4️⃣ TRAINING RESULTS (NÂNG CẤP: TÍNH TOÁN TỰ ĐỘNG)
    # ==================================================
    st.subheader("4️⃣ Training Results (Real-time Calculation)")

    if model and vectorizer:
        # --- Tự động dự đoán và tính điểm ---
        X_test = vectorizer.transform(processed_data["review_clean"])
        y_true = processed_data["label"]
        y_pred = model.predict(X_test)

        # Tính chỉ số thật
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        # Hiển thị bảng kết quả (Đã tính toán)
        results = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Score": [acc, prec, recall_score(y_true, y_pred, average='weighted', zero_division=0), f1]
        })
        st.table(results)

        # --- NÂNG CẤP: VẼ BIỂU ĐỒ CONFUSION MATRIX ---
        st.markdown("**📊 Visualizations**")
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.write("*Confusion Matrix:*")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                        xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot(fig_cm)

        with col_viz2:
            st.write("*WordCloud (Feature Visualization):*")
            text_wc = " ".join(processed_data["review_clean"])
            wc = WordCloud(width=400, height=300, background_color='white', colormap='tab10').generate(text_wc)
            fig_wc, ax_wc = plt.subplots(figsize=(4, 3))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

    st.write("---")

    # ==================================================
    # 5️⃣ MODEL CONFIDENCE (NÂNG CẤP)
    # ==================================================
    st.subheader("5️⃣ Model Confidence Evaluation")

    if model and vectorizer:
        # Lấy xác suất dự đoán (Confidence score)
        probs = model.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        
        confidence_df = pd.DataFrame({
            "Review": processed_data["review"],
            "Predicted": y_pred,
            "Confidence": max_probs
        })
        
        # Format hiển thị màu cho cột Confidence
        st.dataframe(confidence_df.style.background_gradient(subset=["Confidence"], cmap="Greens"))

    st.write("---")

    # ==================================================
    # 6️⃣ CONCLUSION (Giữ nguyên)
    # ==================================================
    st.subheader("6️⃣ Conclusion & Future Work")
    st.markdown(
        """
        **Conclusion:**
        - Model được load trực tiếp và tính toán realtime.
        - Hệ thống tích hợp Visualization (Biểu đồ) giúp dễ dàng đánh giá.

        **Future Work:**
        - Mở rộng dataset.
        - Áp dụng Transformer (BERT, PhoBERT).
        """
    )

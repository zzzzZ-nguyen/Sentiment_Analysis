import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==================================================
# 📦 LOAD MODEL OBJECTS
# ==================================================
@st.cache_resource
def load_model_objects():
    # Đường dẫn tương đối
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
        "<h3 style='color:#2b6f3e;'>Training Info – Sentiment Analysis (Live Calc)</h3>",
        unsafe_allow_html=True
    )

    st.write(
        "Phần này hiển thị thông số huấn luyện thực tế và đánh giá mô hình dựa trên dữ liệu mẫu."
    )
    st.write("---")

    # Load Model
    model, vectorizer = load_model_objects()

    # ==================================================
    # 1️⃣ RAW DATASET
    # ==================================================
    st.subheader("1️⃣ Raw Dataset")

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
    st.write("---")

    # ==================================================
    # 3️⃣ MODEL INFORMATION
    # ==================================================
    st.subheader("3️⃣ Model Information")
    
    if model and vectorizer:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Model:** {type(model).__name__}")
            st.write(f"- Classes: {model.classes_}")
            st.write(f"- Solver: {getattr(model, 'solver', 'N/A')}")
        with col2:
            st.success(f"**Vectorizer:** {type(vectorizer).__name__}")
            st.write(f"- Vocab Size: {len(vectorizer.vocabulary_)} words")
            st.write(f"- N-gram: {vectorizer.ngram_range}")
    else:
        st.error("⚠️ Không tìm thấy file model. Hãy kiểm tra thư mục 'models/'.")

    st.write("---")

    # ==================================================
    # 4️⃣ TRAINING RESULTS (TỰ ĐỘNG TÍNH & VẼ BIỂU ĐỒ STREAMLIT)
    # ==================================================
    st.subheader("4️⃣ Training Results & Visualization")

    if model and vectorizer:
        # --- Tính toán ---
        X_test = vectorizer.transform(processed_data["review_clean"])
        y_true = processed_data["label"]
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Hiển thị số liệu dạng Metric Card (Đẹp hơn bảng)
        m1, m2 = st.columns(2)
        m1.metric("Accuracy (Độ chính xác)", f"{acc*100:.1f}%", delta="Target: >85%")
        m2.metric("F1-Score", f"{f1:.4f}")

        # --- VẼ CONFUSION MATRIX BẰNG DATAFRAME (KHÔNG CẦN MATPLOTLIB) ---
        st.markdown("##### Confusion Matrix (Ma trận nhầm lẫn)")
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
        
        # Tô màu đậm nhạt dựa trên giá trị (Thay thế Heatmap)
        st.dataframe(cm_df.style.background_gradient(cmap="Blues"))
        st.caption("Trục dọc: Thực tế | Trục ngang: Dự đoán")

    st.write("---")

    # ==================================================
    # 5️⃣ MODEL CONFIDENCE (ĐỘ TIN CẬY)
    # ==================================================
    st.subheader("5️⃣ Model Confidence Evaluation")

    if model and vectorizer:
        probs = model.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        
        confidence_df = pd.DataFrame({
            "Review": processed_data["review"],
            "Prediction": y_pred,
            "Confidence": max_probs
        })
        
        # Tô màu xanh cho độ tin cậy cao
        st.dataframe(
            confidence_df.style.background_gradient(subset=["Confidence"], cmap="Greens"),
            use_container_width=True
        )
        
        # Biểu đồ phân bố độ tin cậy (Dùng chart có sẵn của Streamlit)
        st.markdown("##### Phân bố độ tin cậy (Confidence Distribution)")
        st.bar_chart(confidence_df.set_index("Prediction")["Confidence"])

    st.write("---")

    # ==================================================
    # 6️⃣ CONCLUSION
    # ==================================================
    st.subheader("6️⃣ Conclusion")
    st.markdown("""
    * **Hiệu năng:** Model hoạt động ổn định với các câu ngắn.
    * **Tốc độ:** Phản hồi tức thì (Real-time).
    * **Cải tiến:** Giao diện đã được tối ưu hóa hiển thị tự động.
    """)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from wordcloud import WordCloud

# ==================================================
# 📦 LOAD RESOURCES
# ==================================================
@st.cache_resource
def load_resources():
    # Đường dẫn file (Bạn chỉnh lại cho đúng đường dẫn thực tế)
    model_path = os.path.join("models", "model_en.pkl")
    vectorizer_path = os.path.join("models", "vectorizer_en.pkl")
    
    # ⚠️ LƯU Ý: Để tự động cập nhật, bạn nên load file CSV dữ liệu thật
    # Ví dụ: data = pd.read_csv("data/processed_data.csv")
    # Ở đây mình tạo dữ liệu giả lập để code chạy được ngay
    data = pd.DataFrame({
        "review_clean": [
            "good product", "excellent service", "bad quality", "terrible experience", 
            "waste of money", "highly recommend", "average item", "not worth it",
            "very happy", "disappointed", "love it", "hate it", "neutral feeling"
        ],
        "label": [
            "positive", "positive", "negative", "negative", 
            "negative", "positive", "neutral", "negative",
            "positive", "negative", "positive", "negative", "neutral"
        ]
    })

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, data
    except Exception as e:
        return None, None, data

# ==================================================
# 📊 TRAINING INFO FUNCTION
# ==================================================
def show():
    st.markdown("<h2 style='color:#E58E61;'>⚙️ Training Pipeline & Model Evaluation</h2>", unsafe_allow_html=True)
    st.write("Thông tin chi tiết về quá trình huấn luyện, đánh giá hiệu năng và giải thích mô hình.")

    model, vectorizer, data = load_resources()

    if model is None:
        st.error("⚠️ Không tìm thấy file model (.pkl). Vui lòng kiểm tra thư mục 'models/'.")
        return

    # Tách dữ liệu để đánh giá (Trong thực tế nên dùng tập Test riêng)
    X_test = data["review_clean"]
    y_test = data["label"]
    
    # Dự đoán thời gian thực để lấy chỉ số
    X_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_tfidf)

    # --- TABS GIAO DIỆN ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Stats", "📈 Model Performance", "🧠 Feature Importance", "🔍 Model Params"])

    # ==================================================
    # TAB 1: DATASET STATISTICS
    # ==================================================
    with tab1:
        st.subheader("1. Dữ liệu huấn luyện")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Class Distribution:**")
            dist_df = data['label'].value_counts()
            st.dataframe(dist_df, use_container_width=True)
            
            # Biểu đồ tròn phân bố
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(dist_df, labels=dist_df.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99','#ffcc99'])
            st.pyplot(fig)

        with col2:
            st.write("**Word Cloud (Từ khóa phổ biến):**")
            text = " ".join(review for review in data.review_clean)
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
            
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

    # ==================================================
    # TAB 2: PERFORMANCE METRICS (TỰ ĐỘNG TÍNH)
    # ==================================================
    with tab2:
        st.subheader("2. Hiệu năng mô hình (Real-time Calculation)")
        
        # Tính toán metrics
        acc = accuracy_score(y_test, y_pred)
        # Sử dụng average='weighted' vì đây là bài toán đa lớp (3 lớp: pos, neg, neu)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Hiển thị Metrics dạng Card đẹp
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2%}", delta="Goal: >85%")
        m2.metric("Precision", f"{prec:.2%}")
        m3.metric("Recall", f"{rec:.2%}")
        m4.metric("F1-Score", f"{f1:.2%}")

        st.divider()

        # Confusion Matrix
        col_cm1, col_cm2 = st.columns([2, 1])
        with col_cm1:
            st.markdown("##### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
            plt.ylabel('Thực tế')
            plt.xlabel('Dự đoán')
            st.pyplot(fig_cm)
        
        with col_cm2:
            st.info("""
            **Giải thích:**
            - **Đường chéo chính (Màu đậm):** Số lượng dự đoán đúng.
            - **Các ô khác:** Số lượng dự đoán sai.
            - Dữ liệu này được tính toán trực tiếp từ tập dữ liệu tải lên.
            """)

    # ==================================================
    # TAB 3: FEATURE IMPORTANCE (PHẦN XỊN NHẤT)
    # ==================================================
    with tab3:
        st.subheader("3. Mô hình học được gì? (Feature Importance)")
        st.caption("Các từ ngữ ảnh hưởng nhiều nhất đến quyết định của mô hình Logistic Regression.")

        if hasattr(model, 'coef_'):
            # Lấy tên các feature từ vectorizer
            feature_names = vectorizer.get_feature_names_out()
            
            # Lấy hệ số (coefficient) của từng class
            # Giả sử class 'positive' nằm ở index nào đó, ta cần tìm index đó
            classes = model.classes_
            
            # Chọn class để xem
            selected_class = st.selectbox("Chọn nhãn cảm xúc để xem từ khóa đặc trưng:", classes)
            class_index = np.where(classes == selected_class)[0][0]
            
            # Lấy top 10 từ khóa ảnh hưởng nhất
            coefs = model.coef_[class_index]
            
            # Sắp xếp
            top_positive_indices = np.argsort(coefs)[-10:] # Top 10 giá trị lớn nhất (tích cực cho class này)
            top_negative_indices = np.argsort(coefs)[:10]  # Top 10 giá trị nhỏ nhất (tiêu cực cho class này)

            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                st.markdown(f"**Top từ khóa ĐẶC TRƯNG cho '{selected_class}'** (Hệ số cao)")
                top_words = [feature_names[i] for i in top_positive_indices]
                top_scores = coefs[top_positive_indices]
                
                df_top = pd.DataFrame({'Word': top_words, 'Score': top_scores})
                st.bar_chart(df_top.set_index('Word'), color="#2b6f3e")

            with col_f2:
                st.markdown(f"**Top từ khóa CHỐNG LẠI '{selected_class}'** (Hệ số thấp)")
                neg_words = [feature_names[i] for i in top_negative_indices]
                neg_scores = coefs[top_negative_indices]
                
                df_neg = pd.DataFrame({'Word': neg_words, 'Score': neg_scores})
                st.bar_chart(df_neg.set_index('Word'), color="#A20409")

        else:
            st.warning("Mô hình này không hỗ trợ trích xuất Feature Importance (VD: SVM kernel rbf).")

    # ==================================================
    # TAB 4: MODEL PARAMETERS
    # ==================================================
    with tab4:
        st.subheader("4. Thông số kỹ thuật")
        
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("### 📌 Model Configuration")
            st.json({
                "Type": type(model).__name__,
                "Solver": getattr(model, 'solver', 'N/A'),
                "C (Regularization)": getattr(model, 'C', 'N/A'),
                "Max Iterations": getattr(model, 'max_iter', 'N/A'),
                "Classes": list(model.classes_)
            })

        with p2:
            st.markdown("### 📌 Vectorizer Configuration")
            st.json({
                "Type": type(vectorizer).__name__,
                "Vocabulary Size": len(vectorizer.vocabulary_),
                "N-gram Range": vectorizer.ngram_range,
                "Analyzer": vectorizer.analyzer
            })
            
    st.write("---")
    st.caption("© 2025 Auto-generated Report based on loaded `.pkl` models.")

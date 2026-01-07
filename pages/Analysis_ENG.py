import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ==================================================
# 1. CẤU HÌNH TRANG (Bắt buộc đầu tiên)
# ==================================================
st.set_page_config(page_title="Analysis (English)", page_icon="🇬🇧", layout="wide")

# ==================================================
# 2. CSS GIAO DIỆN
# ==================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F0EBD6;
    background-image: repeating-linear-gradient(45deg, #F0EBD6, #F0EBD6 20px, #BBDEA4 20px, #BBDEA4 40px);
}
div.stButton > button {
    background-color: #2b6f3e; color: white; width: 100%; border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. LOAD MODEL (Kết nối Model thật)
# ==================================================
@st.cache_resource
def load_model_objects():
    # Tìm file model trong thư mục models/ hoặc ../models/
    paths = [
        os.path.join("models", "model_en.pkl"),
        os.path.join("..", "models", "model_en.pkl")
    ]
    
    for p in paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                vec_path = p.replace("model_en.pkl", "vectorizer_en.pkl")
                vectorizer = joblib.load(vec_path)
                return model, vectorizer
            except:
                continue
    return None, None

# ==================================================
# 4. GIAO DIỆN CHÍNH
# ==================================================
st.markdown("<h2 style='color:#2b6f3e;'>🇬🇧 English Sentiment Analysis</h2>", unsafe_allow_html=True)
st.write("Analyze product reviews using the trained Logistic Regression model.")

model, vectorizer = load_model_objects()

# Kiểm tra nếu không có model thật thì báo lỗi hoặc dùng Demo tạm
if not model:
    st.warning("⚠️ Could not find 'models/model_en.pkl'. Using a temporary demo model instead.")
    # --- Demo Fallback (Chỉ chạy khi không có file thật) ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    texts = ["Good", "Bad", "Ok"]
    labels = ["positive", "negative", "neutral"]
    vectorizer = TfidfVectorizer()
    X_dummy = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X_dummy, labels)
    # -----------------------------------------------------

# Chia cột: Bên trái nhập liệu đơn, Bên phải upload file
col1, col2 = st.columns([1, 1])

# --- CỘT 1: NHẬP LIỆU ĐƠN ---
with col1:
    st.subheader("📝 Single Review Analysis")
    review = st.text_area("Enter review text:", height=150, placeholder="E.g., The quality is amazing, fast shipping!")

    if st.button("▶️ Analyze Sentiment"):
        if review.strip():
            # Xử lý
            X = vectorizer.transform([review.lower()])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X).max()

            # Hiển thị kết quả đẹp
            st.divider()
            if pred == "positive":
                st.success(f"Prediction: **POSITIVE** (Conf: {proba:.2%})")
            elif pred == "negative":
                st.error(f"Prediction: **NEGATIVE** (Conf: {proba:.2%})")
            else:
                st.info(f"Prediction: **NEUTRAL** (Conf: {proba:.2%})")
        else:
            st.warning("Please enter some text.")

# --- CỘT 2: UPLOAD FILE CSV ---
with col2:
    st.subheader("📂 Batch Analysis (CSV)")
    st.markdown("Upload a CSV file containing a column named **'review'**.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Kiểm tra cột dữ liệu
            # Tự động tìm cột review nếu tên không chuẩn (ví dụ: Comment, text, content)
            target_col = None
            possible_names = ["review", "text", "content", "comment", "description"]
            for col in df.columns:
                if col.lower() in possible_names:
                    target_col = col
                    break
            
            if target_col:
                # Dự đoán hàng loạt
                X_batch = vectorizer.transform(df[target_col].astype(str))
                df["predicted_sentiment"] = model.predict(X_batch)
                
                # Hiển thị bảng kết quả (chỉ 5 dòng đầu)
                st.dataframe(df[[target_col, "predicted_sentiment"]].head(10), use_container_width=True)
                
                # Vẽ biểu đồ
                st.markdown("##### Sentiment Distribution")
                
                # Đếm số lượng
                counts = df["predicted_sentiment"].value_counts()
                
                # Vẽ bằng Matplotlib
                fig, ax = plt.subplots(figsize=(5, 3))
                colors = {'positive': '#66b3ff', 'negative': '#ff9999', 'neutral': '#99ff99'}
                # Map màu cho đúng nhãn
                bar_colors = [colors.get(x, 'gray') for x in counts.index]
                
                counts.plot(kind="bar", ax=ax, color=bar_colors, rot=0)
                plt.ylabel("Count")
                plt.title("Review Sentiment Stats")
                st.pyplot(fig)
                
                # Nút tải kết quả về
                csv_result = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (.csv)",
                    csv_result,
                    "sentiment_results.csv",
                    "text/csv"
                )
                
            else:
                st.error(f"CSV must contain one of these columns: {possible_names}")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.write("---")

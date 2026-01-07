import streamlit as st
import pandas as pd
import joblib
import os

# ==================================================
# 1. HÀM LOAD MODEL (Đã sửa lỗi đường dẫn và cache)
# ==================================================
@st.cache_resource
def load_model_en():
    # Danh sách các vị trí có thể chứa model
    possible_paths = [
        "models/model_en.pkl",       # Chạy từ thư mục gốc
        "../models/model_en.pkl",    # Chạy từ thư mục con
        "pages/models/model_en.pkl"  # Trường hợp khác
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            try:
                # Load Model
                model = joblib.load(p)
                
                # Load Vectorizer (giả sử tên file là vectorizer_en.pkl nằm cùng chỗ)
                vec_path = p.replace("model_en.pkl", "vectorizer_en.pkl")
                if os.path.exists(vec_path):
                    vectorizer = joblib.load(vec_path)
                    return model, vectorizer
            except Exception as e:
                print(f"Lỗi khi load {p}: {e}")
                continue
                
    return None, None

# ==================================================
# 2. GIAO DIỆN CHÍNH (Hàm show)
# ==================================================
def show():
    # --- CSS STYLING ---
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #2b6f3e; color: white; border-radius: 5px; width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    st.markdown("<h2 style='color:#2b6f3e;'>🇬🇧 English Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.write("Enter an English product review to analyze its sentiment (Machine Learning Model).")

    # --- LOAD MODEL ---
    model, vectorizer = load_model_en()

    if model is None:
        st.error("⚠️ Model file not found!")
        st.info("Please make sure you have `model_en.pkl` and `vectorizer_en.pkl` in the `models/` folder.")
        # Dừng chương trình tại đây để không lỗi tiếp
        return

    # --- LAYOUT ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Input Review")
        user_input = st.text_area("Type your review here:", height=150, placeholder="E.g., This product is absolutely amazing!...")
        
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                try:
                    # 1. Vector hóa văn bản
                    vec_text = vectorizer.transform([user_input.lower()])
                    
                    # 2. Dự đoán
                    prediction = model.predict(vec_text)[0]
                    
                    # 3. Tính xác suất (Nếu model hỗ trợ predict_proba)
                    try:
                        proba = model.predict_proba(vec_text).max()
                    except:
                        proba = 1.0 # Mặc định nếu model (như SVM linear) không có proba
                    
                    # 4. Hiển thị kết quả
                    st.write("---")
                    st.markdown("### 🎯 Result")
                    
                    if prediction == "positive" or prediction == 1:
                        st.success(f"**POSITIVE** (Confidence: {proba:.2%})")
                        st.balloons()
                    elif prediction == "negative" or prediction == 0:
                        st.error(f"**NEGATIVE** (Confidence: {proba:.2%})")
                    else:
                        st.warning(f"**NEUTRAL** (Confidence: {proba:.2%})")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.warning("Please enter some text first.")

    with col2:
        st.markdown("### ℹ️ Examples")
        st.info("**Positive:**\n- I love this feature!\n- Highly recommended.")
        st.error("**Negative:**\n- Waste of money.\n- Terrible support.")
        st.warning("**Neutral:**\n- It's okay, not great.\n- Average quality.")

    st.write("---")

# Đoạn này để test chạy độc lập (nếu cần), 
# nhưng khi chạy qua app.py thì nó sẽ gọi hàm show()
if __name__ == "__main__":
    show()

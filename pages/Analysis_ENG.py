import streamlit as st

import pandas as pd

import joblib

import os



# ==================================================

# 1. CẤU HÌNH

# ==================================================

st.set_page_config(page_title="Analysis (English)", page_icon="🇬🇧", layout="wide")



# CSS màu nền

st.markdown("""

<style>

[data-testid="stAppViewContainer"] {

    background-color: #F0EBD6;

    background-image: repeating-linear-gradient(45deg, #F0EBD6, #F0EBD6 20px, #BBDEA4 20px, #BBDEA4 40px);

}

div.stButton > button {

    background-color: #2b6f3e; color: white; border-radius: 5px; width: 100%;

}

</style>

""", unsafe_allow_html=True)



# ==================================================

# 2. LOAD MODEL TIẾNG ANH

# ==================================================

@st.cache_resource

def load_model_en():

    # Tìm file model (thử nhiều đường dẫn để tránh lỗi)

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

# 3. GIAO DIỆN CHÍNH

# ==================================================

st.markdown("<h2 style='color:#2b6f3e;'>🇬🇧 English Sentiment Analysis</h2>", unsafe_allow_html=True)

st.write("Enter an English product review to analyze its sentiment.")



model, vectorizer = load_model_en()



if model is None:

    st.error("⚠️ Model file not found. Please check 'models/model_en.pkl'.")

    st.stop()



# Chia cột cho đẹp

col1, col2 = st.columns([2, 1])



with col1:

    st.markdown("### 📝 Input Review")

    user_input = st.text_area("Type your review here:", height=150, placeholder="E.g., This product is absolutely amazing!...")

    

    if st.button("Analyze Sentiment"):

        if user_input.strip():

            # Dự đoán

            vec_text = vectorizer.transform([user_input.lower()])

            prediction = model.predict(vec_text)[0]

            proba = model.predict_proba(vec_text).max()

            

            # Hiển thị kết quả

            st.write("---")

            st.markdown("### 🎯 Result")

            

            if prediction == "positive":

                st.success(f"**POSITIVE** (Confidence: {proba:.2%})")

                st.balloons()

            elif prediction == "negative":

                st.error(f"**NEGATIVE** (Confidence: {proba:.2%})")

            else:

                st.warning(f"**NEUTRAL** (Confidence: {proba:.2%})")

        else:

            st.warning("Please enter some text first.")



with col2:

    st.markdown("### ℹ️ Examples")

    st.info("**Positive:**\n- I love this feature!\n- Highly recommended.")

    st.error("**Negative:**\n- Waste of money.\n- Terrible support.")

    st.warning("**Neutral:**\n- It's okay, not great.\n- Average quality.")



st.write("---") ,do vẫn bị màn hình trắng và cập nhật thêm lấy dữ liệu từ training infor

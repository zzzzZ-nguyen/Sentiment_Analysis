import streamlit as st
import sys
import os
import random

# Import từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import cả hàm load model VÀ hàm load data
from model_utils import load_model_resources, predict, get_data_files, load_dataset

st.set_page_config(page_title="Analysis", page_icon="🧠", layout="wide")

# CSS
st.markdown("""
<style>
div.stButton > button { background-color: #2b6f3e; color: white; border-radius: 5px; width: 100%; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Deep Learning Sentiment Analysis")

# 1. Load Model
vocab, model = load_model_resources()
if model is None:
    st.error("⚠️ Chưa có Model. Vui lòng Train trước.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Phân tích")
    
    # --- TÍNH NĂNG MỚI: LẤY DỮ LIỆU TỪ FILE ---
    use_sample = st.checkbox("🎲 Lấy câu mẫu từ dữ liệu Training Info")
    
    default_text = ""
    if use_sample:
        files = get_data_files()
        if files:
            # Lấy file đầu tiên hoặc cho user chọn (để đơn giản mình lấy file đầu)
            df = load_dataset(files[0]) 
            if df is not None:
                # Tìm cột chứa chữ (text)
                text_cols = [c for c in df.columns if df[c].dtype == 'object']
                if text_cols:
                    # Lấy ngẫu nhiên 1 dòng
                    random_row = df.sample(1).iloc[0]
                    default_text = str(random_row[text_cols[0]]) # Lấy cột text đầu tiên tìm thấy
                    st.caption(f"Đã lấy từ file `{files[0]}`: {default_text[:50]}...")
    
    # Input Area
    if default_text:
        user_input = st.text_area("Nội dung:", value=default_text, height=150)
    else:
        user_input = st.text_area("Nội dung:", placeholder="Nhập review...", height=150)
    
    if st.button("🚀 Phân tích ngay"):
        if user_input.strip():
            score = predict(user_input, vocab, model)
            
            st.write("---")
            if score >= 0.6:
                st.success(f"**TÍCH CỰC** ({score:.2%})")
            elif score <= 0.4:
                st.error(f"**TIÊU CỰC** ({(1-score):.2%})")
            else:
                st.warning(f"**TRUNG TÍNH** ({score:.2f})")

with col2:
    st.info("💡 **Mẹo:** Tích vào ô 'Lấy câu mẫu' để test nhanh dữ liệu thực tế mà không cần gõ tay.")

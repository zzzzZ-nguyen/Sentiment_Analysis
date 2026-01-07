import streamlit as st
import pandas as pd
import numpy as np

# ==========================
# ⚙️ CẤU HÌNH TRANG (Bắt buộc dòng đầu tiên)
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis for E-Commerce",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# 🎨 CSS STYLING (Giữ nguyên của bạn)
# ==========================
css_style = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F0EBD6;
    background-image: repeating-linear-gradient(45deg, #F0EBD6, #F0EBD6 20px, #BBDEA4 20px, #BBDEA4 40px);
}
[data-testid="stHeader"] { background-color: rgba(255,255,255,0.6); backdrop-filter: blur(5px); }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 3px solid #E58E61; }
div[data-testid="stTable"], div[data-testid="stDataFrame"] { background-color: #ffffff !important; padding: 10px; border-radius: 10px; }
h1, h2, h3 { color: #A20409 !important; }
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# ==========================
# 🎨 HEADER & SIDEBAR
# ==========================
col1, col2 = st.columns([1, 9])
with col1: st.image("https://cdn-icons-png.flaticon.com/512/263/263142.png", width=70)
with col2:
    st.markdown("""
        <h2 style="color:#A20409; margin-bottom:0;">Topic 5: Developing a Sentiment Analysis Application</h2>
        <h4 style="color:#E58E61;">Supporting E-Commerce Business Decision Making</h4>
        """, unsafe_allow_html=True)
st.write("---")

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Go to:", [
    "Home – Giới thiệu đề tài",
    "Training Info – Thông tin mô hình",  # <--- Mới
    "Train PyTorch – Huấn luyện Model",   # <--- Mới
    "Analysis – Sentiment Analysis",
    "Future Scope – Hướng phát triển"
])

# ==========================
# 📦 ROUTING (ĐIỀU HƯỚNG)
# ==========================

if page == "Home – Giới thiệu đề tài":
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("📖 Project Introduction")
    st.info("The project develops an intelligent sentiment analysis system using LSTM & Machine Learning.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- GỌI FILE TRAINING INFO ---
elif page == "Training Info – Thông tin mô hình":
    try:
        from pages import Training_Info
        Training_Info.show()  # Gọi hàm show()
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}. Hãy kiểm tra file `pages/Training_Info.py`")

# --- GỌI FILE TRAIN PYTORCH ---
elif page == "Train PyTorch – Huấn luyện Model":
    try:
        from pages import train_pytorch
        train_pytorch.show()  # Gọi hàm show()
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}. Hãy kiểm tra file `pages/train_pytorch.py`")

# --- CÁC TRANG KHÁC ---
elif page == "Analysis – Sentiment Analysis":
    st.info("Chức năng dự đoán đang được cập nhật...")

elif page == "Future Scope – Hướng phát triển":
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px;"><h3>🚀 Hướng phát triển</h3></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 Student Project Group | Data Science & AI")

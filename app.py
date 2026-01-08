import streamlit as st
import pandas as pd
import numpy as np
import importlib # Thêm thư viện này để reload module

# ==========================
# ⚙️ CẤU HÌNH TRANG (Bắt buộc dòng đầu tiên)
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis for E-Commerce",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 🎨 CSS STYLING
# ==========================
css_style = """
<style>
/* 1. Background Sọc Chéo */
[data-testid="stAppViewContainer"] {
    background-color: #F0EBD6;
    background-image: repeating-linear-gradient(
        45deg,
        #F0EBD6,
        #F0EBD6 20px,
        #BBDEA4 20px,
        #BBDEA4 40px
    );
    background-attachment: fixed;
}

/* 2. Header trong suốt */
[data-testid="stHeader"] {
    background-color: rgba(255,255,255,0.6);
    backdrop-filter: blur(5px);
}

/* 3. Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 3px solid #E58E61;
}

/* 4. TABLE STYLING */
div[data-testid="stTable"], div[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

h1, h2, h3 { color: #A20409 !important; }
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# ==========================
# 🎨 HEADER
# ==========================
col1, col2 = st.columns([1, 9])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/263/263142.png", width=70)

with col2:
    st.markdown(
        """
        <h2 style="color:#A20409; margin-bottom:0; text-shadow: 2px 2px 0px #fff;">
        Topic 5: Developing a Sentiment Analysis Application for Product Reviews
        </h2>
        <h4 style="color:#E58E61; margin-top:4px; text-shadow: 1px 1px 0px #fff;">
        Supporting E-Commerce Business Decision Making (Open-source + Streamlit)
        </h4>
        """,
        unsafe_allow_html=True
    )

st.write("---")

# ==========================
# 📌 SIDEBAR – NAVIGATION
# ==========================
st.sidebar.markdown("## 🧭 Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "Home – Giới thiệu đề tài",
        "EDA – Khám phá dữ liệu",
        "Analysis – Sentiment Analysis",
        "Train PyTorch – Huấn luyện Model",
        "Training Info – Thông tin mô hình",
        "Future Scope – Hướng phát triển"
    ]
)

# ==========================
# 📦 ROUTING (ĐIỀU HƯỚNG NỘI DUNG)
# ==========================

# --- TRANG HOME ---
if page == "Home – Giới thiệu đề tài":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.title("📖 Project Introduction")
        st.info("The project develops an intelligent sentiment analysis system that automatically classifies product reviews into **Positive**, **Neutral**, or **Negative** using LSTM & Machine Learning.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG EDA ---
elif page == "EDA – Khám phá dữ liệu":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("📊 Exploratory Data Analysis (EDA)")
        st.write("Nội dung EDA sẽ hiển thị ở đây.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG ANALYSIS (ĐÃ SỬA LỖI MÀN HÌNH TRẮNG) ---
elif page == "Analysis – Sentiment Analysis":
    try:
        import pages.Analysis
        # Bắt buộc reload để cập nhật code mới nhất từ file Analysis.py
        importlib.reload(pages.Analysis)
        
        # Gọi hàm show()
        pages.Analysis.show()
        
    except ImportError:
        st.warning("⚠️ File `pages/Analysis.py` not found.")
    except AttributeError as e:
        st.error(f"⚠️ Lỗi cấu trúc code: {e}")
        st.info("Hãy chắc chắn file `pages/Analysis.py` đã có hàm `def show():`")
    except Exception as e:
        st.error(f"⚠️ Lỗi không xác định: {e}")

# --- TRANG TRAIN PYTORCH ---
elif page == "Train PyTorch – Huấn luyện Model":
    try:
        from pages import train_pytorch
        train_pytorch.show()
    except Exception as e:
        st.info(f"Đang phát triển module Train: {e}")

# --- TRANG TRAINING INFO ---
elif page == "Training Info – Thông tin mô hình":
    try:
        from pages import Training_Info
        Training_Info.show()
    except Exception as e:
        st.info(f"Đang phát triển module Info: {e}")

# --- TRANG FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("🚀 Hướng phát triển & Kết luận")
        st.write("Nội dung kết luận.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# 👣 FOOTER
# ==========================
st.markdown("---")
_, col_footer, _ = st.columns([1, 8, 1])

with col_footer:
    st.markdown(
        """
        <div style="background: linear-gradient(to right, #E58E61, #e39d7a); border-radius: 12px; padding: 20px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color:white; margin:0;">🎓 Students Group</h4>
            <div style="font-size:15px; line-height:1.6;">
                <b>1. Bui Duc Nguyen</b> - 235053154<br>
                <b>2. Huynh Ngoc Minh Quan</b> - 235052863
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

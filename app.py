import streamlit as st
import pandas as pd
import numpy as np
import importlib # Thư viện để reload module tránh lỗi màn hình trắng

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
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #F0EBD6;
    background-image: repeating-linear-gradient(45deg, #F0EBD6, #F0EBD6 20px, #BBDEA4 20px, #BBDEA4 40px);
    background-attachment: fixed;
}
/* Header */
[data-testid="stHeader"] {
    background-color: rgba(255,255,255,0.6);
    backdrop-filter: blur(5px);
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 3px solid #E58E61;
}
/* Table */
div[data-testid="stTable"], div[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 10px;
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
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Objectives")
            st.markdown("- ✅ Analyze customer opinions\n- ✅ Support Vietnamese/English\n- ✅ Real-time prediction")
        with c2:
            st.markdown("### Technologies")
            st.markdown("- Python, Streamlit\n- Scikit-learn, TF-IDF\n- **PyTorch (LSTM)**")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG EDA ---
elif page == "EDA – Khám phá dữ liệu":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("📊 Exploratory Data Analysis (EDA)")
        st.write("Nội dung phân tích dữ liệu sẽ hiển thị ở đây.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG ANALYSIS (QUAN TRỌNG: CÓ RELOAD) ---
elif page == "Analysis – Sentiment Analysis":
    try:
        from pages import Analysis
        importlib.reload(Analysis) # Reload để cập nhật code mới
        Analysis.show()
    except ImportError:
        st.error("⚠️ Không tìm thấy file `pages/Analysis.py`.")
    except AttributeError:
        st.error("⚠️ File `Analysis.py` thiếu hàm `show()`.")
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")

# --- TRANG TRAIN PYTORCH ---
elif page == "Train PyTorch – Huấn luyện Model":
    try:
        from pages import train_pytorch
        train_pytorch.show()
    except Exception as e:
        st.info("Module đang phát triển.")

# --- TRANG TRAINING INFO ---
elif page == "Training Info – Thông tin mô hình":
    try:
        from pages import Training_Info
        Training_Info.show()
    except Exception as e:
        st.info("Module đang phát triển.")

# --- TRANG FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("🚀 Hướng phát triển")
        st.write("- Mở rộng tập dữ liệu.\n- Áp dụng mô hình BERT/PhoBERT.")
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
        <div style="margin-top: 15px; background: #9BBA74; border-radius: 12px; padding: 15px 20px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
             <h4 style="color:white; margin:0;">👨‍🏫 Instructor: <b>Bùi Tiến Đức</b> –
            <a href="https://orcid.org/0000-0001-5174-3558"
               target="_blank"
               style="text-decoration:none; color:#1a73e8;">
               ORCID: 0000-0001-5174-3558
            </a></h4>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """<div style="text-align:center; margin-top:20px; padding:10px; font-size:13px; color:#A20409; font-weight:bold; background-color: rgba(255,255,255,0.8); border-radius: 20px;">
        © 2025 – Topic 5: Sentiment Analysis for E-Commerce
    </div>""",
    unsafe_allow_html=True
)

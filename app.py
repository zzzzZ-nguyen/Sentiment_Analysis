import streamlit as st
import pandas as pd
import numpy as np

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

/* 4. Thẻ metric đẹp */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #E58E61;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
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
# 🏠 NỘI DUNG TRANG CHỦ (HOME)
# ==========================
# Ở chế độ này, app.py chính là trang Home.
# Các trang khác sẽ tự hiện trên Sidebar.

with st.container():
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("📖 Project Introduction")
    
    st.markdown("### 1. Problem Overview")
    st.info("The project develops an intelligent sentiment analysis system that automatically classifies product reviews into **Positive**, **Neutral**, or **Negative**.")

    col_home1, col_home2 = st.columns(2)
    with col_home1:
        st.markdown("### 2. Objectives")
        st.markdown("""
        * ✅ **Analyze customer opinions** from product reviews.
        * ✅ **Support Vietnamese and English** text.
        * ✅ **Visualize sentiment distribution**.
        * ✅ **Provide real-time sentiment prediction**.
        """)

    with col_home2:
        st.markdown("### 3. Technologies")
        st.markdown("""
        * **Core:** 🐍 Python, 🔴 Streamlit
        * **Processing:** Scikit-learn, TF-IDF, NLTK
        * **Models:** Logistic Regression, SVM, LSTM (PyTorch)
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# 👣 FOOTER
# ==========================
st.markdown("---")

_, col_footer, _ = st.columns([1, 8, 1])

with col_footer:
    st.markdown(
        """
        <div style="background: linear-gradient(to right, #E58E61, #e39d7a); border-radius: 12px; padding: 20px; margin-bottom: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <h4 style="color:white; margin:0; text-transform: uppercase; letter-spacing:1px;">🎓 Students Group</h4>
            </div>
            <div style="font-size:15px; line-height:1.6;">
                <b>1. Bui Duc Nguyen</b> - 235053154<br>
                <b>2. Huynh Ngoc Minh Quan</b> - 235052863
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background: #9BBA74; border-radius: 12px; padding: 15px 20px; color: white; display: flex; align-items: center; gap: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
             <div style="min-width: 120px;">
                <h4 style="color:white; margin:0;">👨‍🏫 Instructor</h4>
            </div>
            <div style="width: 1px; height: 30px; background-color: rgba(255,255,255,0.5);"></div>
            <div style="display: flex; align-items: center; gap: 8px;">
                 <span style="font-weight:bold; font-size: 16px;"> <b>Bùi Tiến Đức</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div style="text-align:center; margin-top:20px; padding:10px; font-size:13px; color:#A20409; font-weight:bold; background-color: rgba(255,255,255,0.8); border-radius: 20px;">
        © 2025 – Topic 5: Sentiment Analysis for E-Commerce
    </div>
    """,
    unsafe_allow_html=True
)

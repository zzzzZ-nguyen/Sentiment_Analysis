import streamlit as st

# ==================================================
# 1. CẤU HÌNH TRANG (Bắt buộc ở dòng đầu tiên)
# ==================================================
st.set_page_config(
    page_title="Topic 5 - Home",
    page_icon="🏠",
    layout="wide"
)

# ==================================================
# 2. CSS & GIAO DIỆN (Đồng bộ với các trang khác)
# ==================================================
st.markdown("""
<style>
    /* Hình nền sọc chéo giống các trang khác */
    [data-testid="stAppViewContainer"] {
        background-color: #F0EBD6;
        background-image: repeating-linear-gradient(45deg, #F0EBD6, #F0EBD6 20px, #BBDEA4 20px, #BBDEA4 40px);
    }
    
    /* Style cho tiêu đề chính */
    .main-title {
        color: #2b6f3e;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Style cho các hộp nội dung (Box) */
    .info-box {
        background-color: #fff7cc;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #e6d784;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
    }
    
    .box-header {
        color: #b30000;
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. NỘI DUNG TRANG CHỦ
# ==================================================

# Tiêu đề chính
st.markdown('<div class="main-title">Topic 5 – Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<h4 style="text-align: center; color: #555;">Product Reviews Classification System</h4>', unsafe_allow_html=True)

st.write("---")

# Chia bố cục thành 2 cột
col1, col2 = st.columns([1, 1])

# --- CỘT TRÁI: Vấn đề & Mục tiêu ---
with col1:
    # Box 1: Problem Overview
    st.markdown("""
    <div class="info-box">
        <div class="box-header">📌 1. Problem Overview</div>
        The project develops an intelligent sentiment analysis system that automatically
        classifies product reviews into <b>Positive, Neutral, or Negative</b>
        to support decision-making for e-commerce businesses.
    </div>
    """, unsafe_allow_html=True)

    # Box 2: Objectives
    st.markdown("""
    <div class="info-box">
        <div class="box-header">🎯 2. Objectives</div>
        <ul>
            <li>Analyze customer opinions from product reviews.</li>
            <li>Support Vietnamese and English text.</li>
            <li>Visualize sentiment distribution.</li>
            <li>Provide real-time sentiment prediction.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- CỘT PHẢI: Công nghệ & Hướng dẫn ---
with col2:
    # Box 3: Technologies
    st.markdown("""
    <div class="info-box">
        <div class="box-header">💻 3. Technologies</div>
        <ul>
            <li><b>Language:</b> Python</li>
            <li><b>Framework:</b> Streamlit (Web App)</li>
            <li><b>Machine Learning:</b> Scikit-learn, TF-IDF</li>
            <li><b>Algorithms:</b> Logistic Regression, SVM</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Box 4: Getting Started (Hướng dẫn nhanh)
    st.markdown("""
    <div class="info-box" style="background-color: #e8f5e9; border-left: 5px solid #2b6f3e;">
        <div class="box-header" style="color: #2b6f3e;">🚀 Getting Started</div>
        <p>Please select a module from the sidebar to begin:</p>
        <ul>
            <li><b>Training Info:</b> View dataset & model metrics.</li>
            <li><b>Analysis:</b> Predict sentiment for new reviews.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.write("---")
st.caption("© 2025 Student Project Group | Data Science & AI")

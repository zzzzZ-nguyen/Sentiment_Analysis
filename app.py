import streamlit as st
import pandas as pd
import numpy as np

# ==========================
# ⚙️ CẤU HÌNH TRANG (Chỉ khai báo 1 lần tại đây)
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 🎨 CSS STYLING
# ==========================
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #F0EBD6;
    background-image: repeating-linear-gradient(45deg, #F0EBD6, #F0EBD6 20px, #E8E4CC 20px, #E8E4CC 40px);
}
/* Header Styles */
h1, h2, h3 { color: #2b6f3e !important; }
/* Sidebar */
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 3px solid #2b6f3e; }
/* Table */
div[data-testid="stTable"], div[data-testid="stDataFrame"] { background-color: white !important; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================
# 🧭 NAVIGATION
# ==========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/263/263142.png", width=80)
st.sidebar.markdown("## 🧭 Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "Home – Giới thiệu",
        "Analysis – Dự đoán (PyTorch)",
        "Training Info – Dữ liệu & Model",
        "Future Scope – Hướng phát triển"
    ]
)

# ==========================
# 📦 ROUTING (ĐIỀU HƯỚNG)
# ==========================

# --- TRANG HOME ---
if page == "Home – Giới thiệu":
    st.title("📖 Project Introduction")
    st.markdown("### Topic 5: Sentiment Analysis for Product Reviews")
    st.info("Hệ thống phân tích cảm xúc đánh giá sản phẩm sử dụng Deep Learning (LSTM) và Machine Learning.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Mục tiêu")
        st.markdown("""
        * ✅ Phân tích ý kiến khách hàng (Positive/Negative/Neutral).
        * ✅ Hỗ trợ Tiếng Việt & Tiếng Anh.
        * ✅ Trực quan hóa dữ liệu huấn luyện.
        """)
    with col2:
        st.subheader("💻 Công nghệ")
        st.markdown("""
        * **Ngôn ngữ:** Python, Streamlit
        * **Deep Learning:** PyTorch (LSTM)
        * **Machine Learning:** Scikit-learn
        """)
    
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*p3_wO5j2h7jQ6bC-uP4u2A.png", caption="Quy trình phân tích cảm xúc")

# --- TRANG ANALYSIS (GỌI FILE CON) ---
elif page == "Analysis – Dự đoán (PyTorch)":
    try:
        from pages.Analysis import show
        show() # Gọi hàm show() từ file Analysis.py
    except ImportError as e:
        st.error(f"❌ Lỗi import: {e}. Hãy đảm bảo file `pages/Analysis.py` tồn tại và có hàm `def show():`")
    except Exception as e:
        st.error(f"❌ Lỗi chạy module: {e}")

# --- TRANG TRAINING INFO (GỌI FILE CON) ---
elif page == "Training Info – Dữ liệu & Model":
    try:
        from pages.Training_Info import show
        show() # Gọi hàm show() từ file Training_Info.py
    except ImportError:
        st.error("❌ Không tìm thấy file `pages/Training_Info.py`.")

# --- TRANG FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    st.header("🚀 Hướng phát triển")
    st.markdown("""
    1. **Mở rộng dữ liệu:** Crawl thêm từ Shopee/Lazada/Tiki.
    2. **Mô hình nâng cao:** Sử dụng BERT/RoBERTa cho tiếng Việt (PhoBERT).
    3. **Triển khai:** Đóng gói thành API thời gian thực.
    """)

# Footer
st.markdown("---")
st.caption("© 2025 Student Project Group | Data Science & AI")

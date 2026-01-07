import streamlit as st
import pandas as pd
import numpy as np

# ==========================
# ⚙️ CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis for E-Commerce",
    page_icon="https://tse4.mm.bing.net/th/id/OIP.ftwMemyVfX2__Kg4dh99wwHaJ3?w=640&h=852&rs=1&pid=ImgDetMain&o=7&rm=3",
    layout="wide"
)

# ==========================
# 🎨 BACKGROUND (SỬ DỤNG 5 MÀU TỪ BẢNG MÀU)
# ==========================
# Tạo background gradient sử dụng các màu từ bảng màu (Kem nhạt -> Xanh bạc hà)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: linear-gradient(to right top, #F0EBD6, #BBDEA4);
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
background-color: #9BBA74; /* Sử dụng màu Xanh ô liu cho Sidebar */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ==========================
# 🎨 HEADER
# ==========================
col1, col2 = st.columns([1, 9])

with col1:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/263/263142.png",
        width=70
    )

with col2:
    st.markdown(
        """
        <h2 style="color:#A20409; margin-bottom:0;"> Topic 5: Developing a Sentiment Analysis Application for Product Reviews
        </h2>
        <h4 style="color:#E58E61; margin-top:4px;"> Supporting E-Commerce Business Decision Making (Open-source + Streamlit)
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
        "Model Comparison – So sánh mô hình",
        "Training Info – Thông tin mô hình",
        "Future Scope – Hướng phát triển"
    ]
)

# ==========================
# 📦 ROUTING (NỘI DUNG CHÍNH)
# ==========================

# --- [CẬP NHẬT] TRANG HOME ---
if page == "Home – Giới thiệu đề tài":
    st.title("📖 Project Introduction")
    
    # Phần 1: Problem Overview
    st.markdown("### 1. Problem Overview")
    st.info(
        "The project develops an intelligent sentiment analysis system that automatically classifies product reviews "
        "into **Positive**, **Neutral**, or **Negative** to support decision-making for e-commerce businesses."
    )

    col_home1, col_home2 = st.columns(2)

    # Phần 2: Objectives
    with col_home1:
        st.markdown("### 2. Objectives")
        st.markdown("""
        * ✅ **Analyze customer opinions** from product reviews.
        * ✅ **Support Vietnamese and English** text.
        * ✅ **Visualize sentiment distribution** (Charts & Graphs).
        * ✅ **Provide real-time sentiment prediction** for new inputs.
        """)

    # Phần 3: Technologies
    with col_home2:
        st.markdown("### 3. Technologies")
        st.markdown("""
        * **Core:** 🐍 Python, 🔴 Streamlit
        * **Processing:** Scikit-learn, TF-IDF
        * **Models:** * 🔹 Logistic Regression
            * 🔹 SVM (Support Vector Machine)
            * 🔹 XGBoost (Optional)
        """)
        
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*p3_wO5j2h7jQ6bC-uP4u2A.png", caption="Sentiment Analysis Workflow Illustration", use_column_width=True)

# --- TRANG EDA ---
elif page == "EDA – Khám phá dữ liệu":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Phân tích sơ bộ về tập dữ liệu đánh giá sản phẩm.")
    
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.subheader("Phân bố nhãn cảm xúc")
        # Giả lập dữ liệu demo
        chart_data = pd.DataFrame({'Sentiment': ['Positive', 'Negative', 'Neutral'], 'Count': [500, 300, 150]})
        st.bar_chart(chart_data.set_index('Sentiment'))
    
    with col_eda2:
        st.subheader("Thống kê từ khóa")
        st.info("Biểu đồ WordCloud hoặc Top Keyword sẽ hiển thị ở đây.")

# --- TRANG ANALYSIS ---
elif page == "Analysis – Sentiment Analysis":
    try:
        from pages.Analysis import show
        show()
    except ImportError:
        st.info("Vui lòng tạo file pages/Analysis.py hoặc thêm code xử lý vào đây.")

# --- TRANG MODEL COMPARISON ---
elif page == "Model Comparison – So sánh mô hình":
    st.header("⚖️ Model Comparison")
    st.markdown("So sánh hiệu quả giữa các thuật toán Machine Learning.")
    
    data = {
        "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
        "Accuracy": ["88%", "85%", "89%", "86%"],
        "F1-Score": ["0.87", "0.84", "0.88", "0.85"],
        "Training Time": ["Low", "Very Low", "High", "Medium"]
    }
    df = pd.DataFrame(data)
    st.table(df)

# --- TRANG TRAINING INFO ---
elif page == "Training Info – Thông tin mô hình":
    try:
        from pages.Training_Info import show
        show()
    except ImportError:
        st.info("Vui lòng tạo file pages/Training_Info.py hoặc thêm code xử lý vào đây.")

# --- TRANG FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    st.header("🚀 Hướng phát triển & Kết luận")
    st.markdown("""
    ### 1. Kết luận
    - Dự án đã xây dựng thành công mô hình phân tích cảm xúc cho E-commerce.
    - Giao diện trực quan hỗ trợ người dùng doanh nghiệp ra quyết định nhanh chóng.

    ### 2. Hạn chế
    - Dữ liệu huấn luyện còn giới hạn.
    - Xử lý ngôn ngữ tự nhiên tiếng Việt phức tạp (teencode, viết tắt).

    ### 3. Hướng phát triển (Future Work)
    - **Mở rộng dữ liệu:** Crawl thêm từ Shopee/Lazada.
    - **Deep Learning:** Áp dụng BERT/RoBERTa.
    """)

# ==========================
# 👣 FOOTER
# ==========================
st.markdown("---")

# -------- STUDENTS BOX (MÀU CAM ĐÀO & ĐỎ THẪM) --------
st.markdown(
    """
    <div style="
        background:#E58E61; /* Sử dụng màu Cam đào làm nền */
        border:2px solid #A20409; /* Sử dụng màu Đỏ thẫm làm viền */
        border-radius:10px;
        padding:16px 20px;
        max-width:900px;
        margin: 0 auto 14px auto;
        font-size:14px;
        line-height:1.7;
        color: #F0EBD6; /* Màu chữ Kem nhạt */
    ">
        <b>Students:</b><br>
        - Bui Duc Nguyen-235053154-nguyenbd23@uef.edu.vn<br>
        - Huynh Ngoc Minh Quan-235052863-quanhnm@uef.edu.vn
    </div>
    """,
    unsafe_allow_html=True
)

# -------- INSTRUCTOR BOX (MÀU XANH Ô LIU & XANH BẠC HÀ) --------
st.markdown(
    """
    <div style="
        background:#9BBA74; /* Sử dụng màu Xanh ô liu làm nền */
        border:2px solid #BBDEA4; /* Sử dụng màu Xanh bạc hà làm viền */
        border-radius:10px;
        padding:14px 20px;
        max-width:900px;
        margin: 0 auto;
        font-size:14px;
        display:flex;
        align-items:center;
        gap:10px;
        color: #F0EBD6; /* Màu chữ Kem nhạt */
    ">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"
             width="22">
        <div>
            <b>Bùi Tiến Đức</b> –
            <a href="https://orcid.org/"
               target="_blank"
               style="text-decoration:none; color:#F0EBD6;"> </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------- COPYRIGHT --------
st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:10px;
        font-size:13px;
        color:#A20409; /* Màu chữ Đỏ thẫm */
    ">
        © 2025 – Topic 5: Sentiment Analysis for E-Commerce
    </div>
    """,
    unsafe_allow_html=True
)

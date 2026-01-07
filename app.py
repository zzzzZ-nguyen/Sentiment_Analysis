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
# 🎨 CSS STYLING
# ==========================
css_style = """
<style>
/* 1. Background Sọc Chéo (Giữ nguyên) */
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

/* 4. TABLE STYLING (CHỈ CÁC BẢNG LÀ MÀU TRẮNG) */
div[data-testid="stTable"], div[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

div[data-testid="stTable"] table, div[data-testid="stDataFrame"] table {
    background-color: #ffffff !important; 
    color: #000000 !important;           
}

/* Header của bảng */
div[data-testid="stTable"] th, div[data-testid="stDataFrame"] th {
    background-color: #E58E61 !important; 
    color: #ffffff !important;           
}

/* Dữ liệu trong bảng */
div[data-testid="stTable"] td, div[data-testid="stDataFrame"] td {
    color: #000000 !important;
}

</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

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
        "Model Comparison – So sánh mô hình",
        "Training Info – Thông tin mô hình",
        "Future Scope – Hướng phát triển"
    ]
)

# ==========================
# 📦 ROUTING (NỘI DUNG CHÍNH)
# ==========================

# --- TRANG HOME ---
if page == "Home – Giới thiệu đề tài":
    with st.container():
        # Container màu trắng mờ để nội dung dễ đọc trên nền sọc
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        
        st.title("📖 Project Introduction")
        
        # Phần 1: Problem Overview
        st.markdown("### 1. Problem Overview")
        st.info(
            "The project develops an intelligent sentiment analysis system that automatically classifies product reviews "
            "into **Positive**, **Neutral**, or **Negative** to support decision-making for e-commerce businesses."
        )

        col_home1, col_home2 = st.columns(2)

        # Phần 2: Objectives (ĐÃ KHÔI PHỤC DẤU CHẤM ĐẦU DÒNG)
        with col_home1:
            st.markdown("### 2. Objectives")
            st.markdown("""
            * ✅ **Analyze customer opinions** from product reviews.
            * ✅ **Support Vietnamese and English** text.
            * ✅ **Visualize sentiment distribution** (Charts & Graphs).
            * ✅ **Provide real-time sentiment prediction** for new inputs.
            """)

        # Phần 3: Technologies (ĐÃ KHÔI PHỤC DẤU CHẤM ĐẦU DÒNG)
        with col_home2:
            st.markdown("### 3. Technologies")
            st.markdown("""
            * **Core:** 🐍 Python, 🔴 Streamlit
            * **Processing:** Scikit-learn, TF-IDF
            * **Models:**
                * 🔹 Logistic Regression
                * 🔹 SVM (Support Vector Machine)
                * 🔹 XGBoost (Optional)
            """)
            
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*p3_wO5j2h7jQ6bC-uP4u2A.png", caption="Sentiment Analysis Workflow Illustration", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG EDA ---
elif page == "EDA – Khám phá dữ liệu":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("📊 Exploratory Data Analysis (EDA)")
        st.markdown("Phân tích sơ bộ về tập dữ liệu đánh giá sản phẩm.")
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.subheader("Phân bố nhãn cảm xúc")
            chart_data = pd.DataFrame({'Sentiment': ['Positive', 'Negative', 'Neutral'], 'Count': [500, 300, 150]})
            st.bar_chart(chart_data.set_index('Sentiment'))
        
        with col_eda2:
            st.subheader("Thống kê từ khóa")
            st.info("Biểu đồ WordCloud hoặc Top Keyword sẽ hiển thị ở đây.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG ANALYSIS ---
elif page == "Analysis – Sentiment Analysis":
    try:
        from pages.Analysis import show
        show()
    except ImportError:
        st.info("Vui lòng tạo file pages/Analysis.py hoặc thêm code xử lý vào đây.")

# --- TRANG MODEL COMPARISON ---
elif page == "Model Comparison – So sánh mô hình":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("⚖️ Model Comparison")
        st.markdown("So sánh hiệu quả giữa các thuật toán Machine Learning.")
        
        data = {
            "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
            "Accuracy": ["88%", "85%", "89%", "86%"],
            "F1-Score": ["0.87", "0.84", "0.88", "0.85"],
            "Training Time": ["Low", "Very Low", "High", "Medium"]
        }
        df = pd.DataFrame(data)
        
        # Bảng này sẽ có màu trắng do CSS
        st.table(df)
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG TRAINING INFO ---
elif page == "Training Info – Thông tin mô hình":
    try:
        from pages.Training_Info import show
        show()
    except ImportError:
        st.info("Vui lòng tạo file pages/Training_Info.py hoặc thêm code xử lý vào đây.")

# --- TRANG FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# 👣 FOOTER (MÀU CŨ - COLORFUL)
# ==========================
st.markdown("---")

# Giữ khung màu Cam/Đỏ và Xanh Olive như yêu cầu
_, col_footer, _ = st.columns([1, 8, 1])

with col_footer:
    # -------- STUDENTS BOX (MÀU CAM ĐÀO & ĐỎ THẪM) --------
    st.markdown(
        """
        <div style="
            background:#E58E61; 
            border:2px solid #A20409; 
            border-radius:10px;
            padding:16px 20px;
            margin-bottom: 14px;
            font-size:14px;
            line-height:1.7;
            color: #F0EBD6; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h4 style="color:#A20409; margin-top:0; margin-bottom:5px;">🎓 Students Group</h4>
            <b>1. Bui Duc Nguyen</b> - 235053154 - nguyenbd23@uef.edu.vn<br>
            <b>2. Huynh Ngoc Minh Quan</b> - 235052863 - quanhnm@uef.edu.vn
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------- INSTRUCTOR BOX (MÀU XANH Ô LIU & XANH BẠC HÀ) --------
    st.markdown(
        """
        <div style="
            background:#9BBA74; 
            border:2px solid #BBDEA4; 
            border-radius:10px;
            padding:14px 20px;
            font-size:14px;
            display:flex;
            align-items:center;
            gap:15px;
            color: #F0EBD6;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
             <div>
                <h4 style="color:#F0EBD6; margin:0; text-shadow: 1px 1px 0 #556B2F;">👨‍🏫 Instructor</h4>
            </div>
            <div style="flex-grow:1; border-left:1px solid #BBDEA4; padding-left:15px;">
                <div style="display:flex; align-items:center; gap:8px;">
                     <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="20">
                     <span style="font-weight:bold;">Bùi Tiến Đức</span>
                </div>
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
        margin-top:20px;
        padding:10px;
        font-size:13px;
        color:#A20409;
        font-weight:bold;
        background-color: rgba(255,255,255,0.8);
        border-radius: 20px;
        display: inline-block;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
    ">
        © 2025 – Topic 5: Sentiment Analysis for E-Commerce
    </div>
    """,
    unsafe_allow_html=True
)

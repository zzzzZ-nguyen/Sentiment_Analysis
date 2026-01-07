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
# 🎨 CSS STYLING (BACKGROUND SỌC, CARD & TABLE)
# ==========================
# Bảng màu: #A20409 (Đỏ), #E58E61 (Cam), #F0EBD6 (Kem), #BBDEA4 (Xanh nhạt), #9BBA74 (Xanh Olive)

css_style = """
<style>
/* 1. Background Sọc Chéo (Striped Gradient) */
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

/* 2. Làm trong suốt Header để thấy nền */
[data-testid="stHeader"] {
    background-color: rgba(255,255,255,0.6);
    backdrop-filter: blur(5px);
}

/* 3. Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 3px solid #E58E61;
}

/* 4. CSS Class cho Khung Tên (Card Style) */
.info-card {
    background-color: rgba(255, 255, 255, 0.95); /* Nền trắng hơi trong */
    border-radius: 15px;       /* Bo tròn góc */
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1); /* Đổ bóng 3D */
    font-family: sans-serif;
    color: #333;
}
.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}

/* 5. TABLE STYLING (Đồng bộ bảng màu trắng) */
/* Áp dụng cho st.table (Model Comparison, Training Info...) */
div[data-testid="stTable"] > table {
    background-color: #ffffff !important; /* Nền trắng tuyệt đối */
    color: #333333 !important;           /* Chữ màu đen xám */
    border-radius: 10px;                 /* Bo góc bảng */
    overflow: hidden;
    border: 1px solid #ddd;              /* Viền nhẹ */
}

/* Header của bảng (Hàng tiêu đề) */
div[data-testid="stTable"] th {
    background-color: #E58E61 !important; /* Màu Cam đào làm nền header */
    color: white !important;              /* Chữ trắng */
    font-weight: bold;
}

/* Dòng chẵn lẻ (Optional: giúp dễ nhìn hơn) */
div[data-testid="stTable"] tr:nth-child(even) {
    background-color: #f9f9f9 !important;
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
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        
        st.title("📖 Project Introduction")
        
        st.markdown("### 1. Problem Overview")
        st.info(
            "The project develops an intelligent sentiment analysis system that automatically classifies product reviews "
            "into **Positive**, **Neutral**, or **Negative** to support decision-making for e-commerce businesses."
        )

        col_home1, col_home2 = st.columns(2)

        with col_home1:
            st.markdown("### 2. Objectives")
            st.markdown("""
            * ✅ **Analyze customer opinions** from product reviews.
            * ✅ **Support Vietnamese and English** text.
            * ✅ **Visualize sentiment distribution** (Charts & Graphs).
            * ✅ **Provide real-time sentiment prediction** for new inputs.
            """)

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
        
        # Dữ liệu mẫu
        data = {
            "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
            "Accuracy": ["88%", "85%", "89%", "86%"],
            "F1-Score": ["0.87", "0.84", "0.88", "0.85"],
            "Training Time": ["Low", "Very Low", "High", "Medium"]
        }
        df = pd.DataFrame(data)
        
        # Hiển thị bảng (CSS phía trên sẽ tự động làm bảng này màu trắng)
        st.table(df)
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG TRAINING INFO ---
elif page == "Training Info – Thông tin mô hình":
    # CSS cũng sẽ tự động áp dụng cho bảng trong file Training_Info.py nếu bạn dùng st.table()
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
# 👣 FOOTER (CARD STYLE)
# ==========================
st.markdown("---")

_, col_footer, _ = st.columns([1, 8, 1])

with col_footer:
    # -------- STUDENTS BOX --------
    st.markdown(
        """
        <div class="info-card" style="border-left: 10px solid #A20409;">
            <h4 style="color:#A20409; margin-top:0;">🎓 Students Group</h4>
            <div style="color:#555;">
                <b>1. Bui Duc Nguyen</b> - 235053154 - nguyenbd23@uef.edu.vn<br>
                <b>2. Huynh Ngoc Minh Quan</b> - 235052863 - quanhnm@uef.edu.vn
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------- INSTRUCTOR BOX --------
    st.markdown(
        """
        <div class="info-card" style="border-left: 10px solid #9BBA74; display:flex; align-items:center; gap:15px;">
             <div>
                <h4 style="color:#9BBA74; margin:0;">👨‍🏫 Instructor</h4>
            </div>
            <div style="flex-grow:1; border-left:1px solid #ddd; padding-left:15px;">
                <div style="display:flex; align-items:center; gap:8px;">
                     <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="20">
                     <span style="font-weight:bold; color:#333;">Bùi Tiến Đức</span>
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
        background-color: rgba(255,255,255,0.6);
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

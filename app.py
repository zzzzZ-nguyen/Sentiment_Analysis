import streamlit as st
import numpy as np

# ==========================
# ⚙️ CẤU HÌNH TRANG
# ⚙️ CẤU HÌNH TRANG (DUY NHẤT TẠI ĐÂY)
# ⚙️ CẤU HÌNH TRANG (Bắt buộc dòng đầu tiên)
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis for E-Commerce",
    page_icon="https://tse4.mm.bing.net/th/id/OIP.ftwMemyVfX2__Kg4dh99wwHaJ3?w=640&h=852&rs=1&pid=ImgDetMain&o=7&rm=3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
    layout="wide"
)

# ==========================
# 🎨 CSS STYLING
# 🎨 CSS STYLING (Giữ nguyên của bạn)
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

/* Header của bảng */
div[data-testid="stTable"] th, div[data-testid="stDataFrame"] th {
    background-color: #f8f9fa !important;
    color: #333333 !important;
    border-bottom: 2px solid #E58E61 !important;
    font-weight: bold;
}

/* Dữ liệu trong bảng */
div[data-testid="stTable"] td, div[data-testid="stDataFrame"] td {
    color: #333333 !important;
    border-bottom: 1px solid #eee !important;
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
# 🎨 HEADER
# 🎨 HEADER & SIDEBAR
# ==========================
col1, col2 = st.columns([1, 9])

with col1:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/263/263142.png",
        width=70
    )
    st.image("https://cdn-icons-png.flaticon.com/512/263/263142.png", width=70)

with col1: st.image("https://cdn-icons-png.flaticon.com/512/263/263142.png", width=70)
with col2:
    st.markdown(
        """
        <h2 style="color:#2b6f3e; margin-bottom:0;">
        <h2 style="color:#A20409; margin-bottom:0; text-shadow: 2px 2px 0px #fff;">
        Topic 5: Developing a Sentiment Analysis Application for Product Reviews
        </h2>
        <h4 style="color:#555; margin-top:4px;">
        <h4 style="color:#E58E61; margin-top:4px; text-shadow: 1px 1px 0px #fff;">
        Supporting E-Commerce Business Decision Making (Open-source + Streamlit)
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <h2 style="color:#A20409; margin-bottom:0;">Topic 5: Developing a Sentiment Analysis Application</h2>
        <h4 style="color:#E58E61;">Supporting E-Commerce Business Decision Making</h4>
        """, unsafe_allow_html=True)
st.write("---")

# ==========================
@@ -44,93 +111,170 @@
    "Go to:",
    [
        "Home – Giới thiệu đề tài",
        "EDA – Khám phá dữ liệu",
        "Analysis – Sentiment Analysis",
        "Training Info – Thông tin mô hình"
        "Model Comparison – So sánh mô hình",
        "Training Info – Thông tin mô hình",
        "Future Scope – Hướng phát triển"
    ]
)
page = st.sidebar.radio("Go to:", [
    "Home – Giới thiệu đề tài",
    "Training Info – Thông tin mô hình",  # <--- Mới
    "Train PyTorch – Huấn luyện Model",   # <--- Mới
    "Analysis – Sentiment Analysis",
    "Future Scope – Hướng phát triển"
])

# ==========================
# 📦 ROUTING
# 📦 ROUTING (ĐIỀU HƯỚNG NỘI DUNG)
# 📦 ROUTING (ĐIỀU HƯỚNG)
# ==========================

# --- TRANG HOME ---
if page == "Home – Giới thiệu đề tài":
    from pages.Home import show
    show()
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.title("📖 Project Introduction")
        
        st.markdown("### 1. Problem Overview")
        st.info("The project develops an intelligent sentiment analysis system that automatically classifies product reviews into **Positive**, **Neutral**, or **Negative**.")
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("📖 Project Introduction")
    st.info("The project develops an intelligent sentiment analysis system using LSTM & Machine Learning.")
    st.markdown('</div>', unsafe_allow_html=True)

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
            * **Processing:** Scikit-learn, TF-IDF
            * **Models:** Logistic Regression, SVM, LSTM (PyTorch)
            """)
            
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*p3_wO5j2h7jQ6bC-uP4u2A.png", caption="Sentiment Analysis Workflow", use_column_width=True)
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

# --- TRANG ANALYSIS (GỌI FILE ANALYSIS.PY) ---
elif page == "Analysis – Sentiment Analysis":
    from pages.Analysis import show
    show()
# --- GỌI FILE TRAINING INFO ---
elif page == "Training Info – Thông tin mô hình":
    try:
        from pages import Analysis
        Analysis.show()  # Gọi hàm show() trong file Analysis.py
    except ImportError:
        st.error("⚠️ Không tìm thấy file `pages/Analysis.py` hoặc hàm `show()`. Vui lòng kiểm tra lại.")
        from pages import Training_Info
        Training_Info.show()  # Gọi hàm show()
    except Exception as e:
        st.error(f"⚠️ Lỗi khi chạy Analysis: {e}")

# --- TRANG MODEL COMPARISON ---
elif page == "Model Comparison – So sánh mô hình":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("⚖️ Model Comparison")
        data = {
            "Model": ["Logistic Regression", "Naive Bayes", "SVM", "LSTM (PyTorch)"],
            "Accuracy": ["88%", "85%", "89%", "92%"],
            "F1-Score": ["0.87", "0.84", "0.88", "0.91"],
            "Training Time": ["Low", "Very Low", "High", "High"]
        }
        st.table(pd.DataFrame(data))
        st.markdown('</div>', unsafe_allow_html=True)
        st.error(f"⚠️ Lỗi: {e}. Hãy kiểm tra file `pages/Training_Info.py`")

# --- TRANG TRAINING INFO (GỌI FILE TRAINING_INFO.PY) ---
elif page == "Training Info – Thông tin mô hình":
    from pages.Training_Info import show
    show()
# --- GỌI FILE TRAIN PYTORCH ---
elif page == "Train PyTorch – Huấn luyện Model":
    try:
        from pages import Training_Info
        Training_Info.show() # Gọi hàm show() trong file Training_Info.py
    except ImportError:
        st.error("⚠️ Không tìm thấy file `pages/Training_Info.py` hoặc hàm `show()`. Vui lòng kiểm tra lại.")
        from pages import train_pytorch
        train_pytorch.show()  # Gọi hàm show()
    except Exception as e:
        st.error(f"⚠️ Lỗi khi chạy Training Info: {e}")
        st.error(f"⚠️ Lỗi: {e}. Hãy kiểm tra file `pages/train_pytorch.py`")

# --- TRANG FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("🚀 Hướng phát triển & Kết luận")
        st.markdown("""
        ### 1. Kết luận
        - Dự án đã xây dựng thành công mô hình phân tích cảm xúc cho E-commerce.
        - Tích hợp Deep Learning (LSTM) cho độ chính xác cao.
# --- CÁC TRANG KHÁC ---
elif page == "Analysis – Sentiment Analysis":
    st.info("Chức năng dự đoán đang được cập nhật...")

        ### 2. Hướng phát triển (Future Work)
        - **Mở rộng dữ liệu:** Crawl thêm từ Shopee/Lazada.
        - **Model:** Áp dụng BERT/RoBERTa (PhoBERT) để xử lý tiếng Việt tốt hơn.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
elif page == "Future Scope – Hướng phát triển":
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px;"><h3>🚀 Hướng phát triển</h3></div>', unsafe_allow_html=True)

# ==========================
# 👣 FOOTER (MATCH IMAGE UI)
# 👣 FOOTER
# ==========================
# Footer
st.markdown("---")
_, col_footer, _ = st.columns([1, 8, 1])

# -------- STUDENTS BOX (YELLOW) --------
st.markdown(
    """
    <div style="
        background:#fffbd6;
        border:1px solid #f0d878;
        border-radius:10px;
        padding:16px 20px;
        max-width:900px;
        margin: 0 auto 14px auto;
        font-size:14px;
        line-height:1.7;
    ">
        <b>Students:</b><br>
        - Bui Duc Nguyen-235053154-nguyenbd23@uef.edu.vn
        - Huynh Ngoc Minh Quan-235052863-quanhnm@uef.edu.vn
    </div>
    """,
    unsafe_allow_html=True
)

# -------- INSTRUCTOR BOX (GRAY) --------
st.markdown(
    """
    <div style="
        background:#f8f9fa;
        border:1px solid #ddd;
        border-radius:10px;
        padding:14px 20px;
        max-width:900px;
        margin: 0 auto;
        font-size:14px;
        display:flex;
        align-items:center;
        gap:10px;
    ">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"
             width="22">
        <div>
            <b>Bùi Tiến Đức</b> –
            <a href="https://orcid.org/0000-0001-5174-3558"
               target="_blank"
               style="text-decoration:none; color:#1a73e8;">
               ORCID: 0000-0001-5174-3558
            </a>
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
    </div>
    """,
    unsafe_allow_html=True
)
        <div style="margin-top: 15px; background: #9BBA74; border-radius: 12px; padding: 15px 20px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
             <h4 style="color:white; margin:0;">👨‍🏫 Instructor: Bùi Tiến Đức</h4>
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
        color:#666;
    ">
    """<div style="text-align:center; margin-top:20px; padding:10px; font-size:13px; color:#A20409; font-weight:bold; background-color: rgba(255,255,255,0.8); border-radius: 20px;">
        © 2025 – Topic 5: Sentiment Analysis for E-Commerce
    </div>
    """,
    </div>""",
    unsafe_allow_html=True

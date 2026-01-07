import streamlit as st
import pandas as pd
import numpy as np

# ==========================
# ⚙️ CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis for E-Commerce",
    page_icon="🧠",
    layout="wide"
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

div[data-testid="stTable"] th, div[data-testid="stDataFrame"] th {
    background-color: #f8f9fa !important;
    color: #333333 !important;
    border-bottom: 2px solid #E58E61 !important;
    font-weight: bold;
}

div[data-testid="stTable"] td, div[data-testid="stDataFrame"] td {
    color: #333333 !important;
    border-bottom: 1px solid #eee !important;
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
            * **Processing:** Scikit-learn, TF-IDF
            * **Models:** Logistic Regression, SVM
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG EDA ---
elif page == "EDA – Khám phá dữ liệu":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("📊 Exploratory Data Analysis (EDA)")
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
        from pages import Analysis
        Analysis.show()
    except ImportError:
        st.error("⚠️ Không tìm thấy file `pages/Analysis.py` hoặc hàm `show()`. Vui lòng kiểm tra lại.")
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")

# --- TRANG MODEL COMPARISON ---
elif page == "Model Comparison – So sánh mô hình":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("⚖️ Model Comparison")
        data = {
            "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
            "Accuracy": ["88%", "85%", "89%", "86%"],
            "F1-Score": ["0.87", "0.84", "0.88", "0.85"],
            "Training Time": ["Low", "Very Low", "High", "Medium"]
        }
        st.table(pd.DataFrame(data))
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG TRAINING INFO ---
import streamlit as st
import pandas as pd
import numpy as np

# ==========================
# ⚙️ CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis for E-Commerce",
    page_icon="🧠",
    layout="wide"
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

div[data-testid="stTable"] th, div[data-testid="stDataFrame"] th {
    background-color: #f8f9fa !important;
    color: #333333 !important;
    border-bottom: 2px solid #E58E61 !important;
    font-weight: bold;
}

div[data-testid="stTable"] td, div[data-testid="stDataFrame"] td {
    color: #333333 !important;
    border-bottom: 1px solid #eee !important;
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
            * **Processing:** Scikit-learn, TF-IDF
            * **Models:** Logistic Regression, SVM
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TRANG EDA ---
elif page == "EDA – Khám phá dữ liệu":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("📊 Exploratory Data Analysis (EDA)")
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
        from pages import Analysis
        Analysis.show()
    except ImportError:
        st.error("⚠️ Không tìm thấy file `pages/Analysis.py` hoặc hàm `show()`. Vui lòng kiểm tra lại.")
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")

# --- TRANG MODEL COMPARISON ---
elif page == "Model Comparison – So sánh mô hình":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("⚖️ Model Comparison")
        data = {
            "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
            "Accuracy": ["88%", "85%", "89%", "86%"],
            "F1-Score": ["0.87", "0.84", "0.88", "0.85"],
            "Training Time": ["Low", "Very Low", "High", "Medium"]
        }
        st.table(pd.DataFrame(data))
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
        ### 2. Hướng phát triển (Future Work)
        - **Mở rộng dữ liệu:** Crawl thêm từ Shopee/Lazada.
        - **Deep Learning:** Áp dụng BERT/RoBERTa.
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

# --- TRANG FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    with st.container():
        st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
        st.header("🚀 Hướng phát triển & Kết luận")
        st.markdown("""
        ### 1. Kết luận
        - Dự án đã xây dựng thành công mô hình phân tích cảm xúc cho E-commerce.
        ### 2. Hướng phát triển (Future Work)
        - **Mở rộng dữ liệu:** Crawl thêm từ Shopee/Lazada.
        - **Deep Learning:** Áp dụng BERT/RoBERTa.
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

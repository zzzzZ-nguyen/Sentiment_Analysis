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
# 🎨 BACKGROUND (MỚI THÊM)
# ==========================
# Tạo background gradient nhẹ nhàng (Xanh mint nhạt -> Trắng)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: linear-gradient(to right top, #e8f5e9, #f1f8e9, #ffffff);
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ==========================
# 🎨 HEADER (CODE CŨ)
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
        <h2 style="color:#2b6f3e; margin-bottom:0;">
        Topic 5: Developing a Sentiment Analysis Application for Product Reviews
        </h2>
        <h4 style="color:#555; margin-top:4px;">
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

# Cập nhật thêm 3 phần mới vào danh sách
page = st.sidebar.radio(
    "Go to:",
    [
        "Home – Giới thiệu đề tài",
        "EDA – Khám phá dữ liệu",           # [MỚI 1]
        "Analysis – Sentiment Analysis",
        "Model Comparison – So sánh mô hình", # [MỚI 2]
        "Training Info – Thông tin mô hình",
        "Future Scope – Hướng phát triển"     # [MỚI 3]
    ]
)

# ==========================
# 📦 ROUTING
# ==========================

# --- TRANG CŨ ---
if page == "Home – Giới thiệu đề tài":
    try:
        from pages.Home import show
        show()
    except ImportError:
        st.info("Đang hiển thị trang Home (Vui lòng tạo file pages/Home.py để ẩn thông báo này)")
        st.markdown("### Xin chào! Đây là trang giới thiệu đề tài.")

# --- [MỚI 1] EDA ---
elif page == "EDA – Khám phá dữ liệu":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Phân tích sơ bộ về tập dữ liệu đánh giá sản phẩm.")
    
    # Demo chart (Bạn có thể thay bằng dữ liệu thật)
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.subheader("Phân bố nhãn cảm xúc")
        # Giả lập dữ liệu
        chart_data = pd.DataFrame({'Sentiment': ['Positive', 'Negative', 'Neutral'], 'Count': [500, 300, 150]})
        st.bar_chart(chart_data.set_index('Sentiment'))
    
    with col_eda2:
        st.subheader("Thống kê từ khóa")
        st.info("Tại đây sẽ hiển thị WordCloud hoặc Top từ khóa xuất hiện nhiều nhất.")

# --- TRANG CŨ ---
elif page == "Analysis – Sentiment Analysis":
    try:
        from pages.Analysis import show
        show()
    except ImportError:
        st.info("Vui lòng tạo file pages/Analysis.py")

# --- [MỚI 2] MODEL COMPARISON ---
elif page == "Model Comparison – So sánh mô hình":
    st.header("⚖️ Model Comparison")
    st.markdown("So sánh hiệu quả giữa các thuật toán Machine Learning.")
    
    # Bảng so sánh giả định
    data = {
        "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
        "Accuracy": ["88%", "85%", "89%", "86%"],
        "F1-Score": ["0.87", "0.84", "0.88", "0.85"],
        "Training Time": ["Low", "Very Low", "High", "Medium"]
    }
    df = pd.DataFrame(data)
    st.table(df)
    st.success("Nhận xét: SVM cho kết quả tốt nhất nhưng tốn nhiều thời gian huấn luyện hơn.")

# --- TRANG CŨ ---
elif page == "Training Info – Thông tin mô hình":
    try:
        from pages.Training_Info import show
        show()
    except ImportError:
        st.info("Vui lòng tạo file pages/Training_Info.py")

# --- [MỚI 3] FUTURE SCOPE ---
elif page == "Future Scope – Hướng phát triển":
    st.header("🚀 Hướng phát triển & Kết luận")
    st.markdown("""
    ### 1. Kết luận
    - Dự án đã xây dựng thành công mô hình phân tích cảm xúc cho E-commerce.
    - Giao diện trực quan hỗ trợ người dùng doanh nghiệp ra quyết định nhanh chóng.

    ### 2. Hạn chế
    - Dữ liệu huấn luyện còn giới hạn ở một số ngành hàng cụ thể.
    - Chưa xử lý tốt các câu văn mang tính châm biếm (sarcasm).

    ### 3. Hướng phát triển (Future Work)
    - **Mở rộng dữ liệu:** Thu thập thêm comment từ Shopee/Lazada thời gian thực.
    - **Deep Learning:** Áp dụng mô hình BERT/RoBERTa để tăng độ chính xác.
    - **Đa ngôn ngữ:** Hỗ trợ phân tích cả Tiếng Anh và Tiếng Việt lẫn lộn.
    """)

# ==========================
# 👣 FOOTER (MATCH IMAGE UI - GIỮ NGUYÊN)
# ==========================
st.markdown("---")

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
        - Bui Duc Nguyen-235053154-nguyenbd23@uef.edu.vn<br>
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
            <a href="https://orcid.org/"
               target="_blank"
               style="text-decoration:none; color:#1a73e8;">
            </a>
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
        color:#666;
    ">
        © 2025 – Topic 5: Sentiment Analysis for E-Commerce
    </div>
    """,
    unsafe_allow_html=True
)

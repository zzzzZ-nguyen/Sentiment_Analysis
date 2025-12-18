import streamlit as st

# ==========================
# ⚙️ CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis for E-Commerce",
    page_icon="🛒",
    layout="wide"
)

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

page = st.sidebar.radio(
    "Go to:",
    [
        "Home – Giới thiệu đề tài",
        "Analysis – Sentiment Analysis",
        "Training Info – Thông tin mô hình"
    ]
)

# ==========================
# 📦 ROUTING
# ==========================
if page == "Home – Giới thiệu đề tài":
    from pages.Home import show
    show()

elif page == "Analysis – Sentiment Analysis":
    from pages.Analysis import show
    show()

elif page == "Training Info – Thông tin mô hình":
    from pages.Training_Info import show
    show()

# ==========================
# 👣 FOOTER (MATCH IMAGE UI)
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

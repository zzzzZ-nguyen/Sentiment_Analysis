import streamlit as st

def show():
    # ==========================================
    # 1. PROBLEM OVERVIEW (Bối cảnh & Vấn đề)
    # ==========================================
    st.markdown("### 1. Problem Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            <div style="text-align: justify;">
            In the rapidly expanding digital economy, e-commerce platforms generate massive amounts of 
            <b>unstructured data</b> in the form of customer product reviews. 
            <br><br>
            For businesses, manually analyzing thousands of reviews to understand customer satisfaction is:
            <ul>
                <li>❌ <b>Time-consuming:</b> Impossible to scale with human effort alone.</li>
                <li>❌ <b>Expensive:</b> High operational costs for manual labeling.</li>
                <li>❌ <b>Prone to Error:</b> Subjective bias in human interpretation.</li>
            </ul>
            This creates a <i>"data-rich, information-poor"</i> scenario where valuable insights into product quality are lost.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # Minh họa đơn giản cho vấn đề quá tải dữ liệu
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2920/2920349.png", 
            caption="Information Overload",
            width=200
        )

    st.markdown("---")

    # ==========================================
    # 2. OBJECTIVES (Mục tiêu đề tài)
    # ==========================================
    st.markdown("### 2. Objectives")
    
    st.info(
        """
        **The primary goal is to develop a lightweight, bilingual Sentiment Analysis Application.**
        """
    )
    
    st.markdown(
        """
        To address the problem above, this project focuses on the following key objectives:
        
        * ✅ **Automated Classification:** Instantly categorize feedback into **Positive**, **Neutral**, or **Negative**.
        * ✅ **Bilingual Support:** Handle both **English** (Global products) and **Vietnamese** (Local market) reviews effectively.
        * ✅ **Real-time Inference:** Provide immediate results for user input via a web interface.
        * ✅ **Decision Support:** Help businesses identify product flaws and improve customer service based on data.
        """
    )

    st.markdown("---")

    # ==========================================
    # 3. PROPOSED METHODOLOGY & TECHNOLOGIES
    # ==========================================
    st.markdown("### 3. Technologies & Methodology")
    
    st.markdown("This system utilizes a **Hybrid Approach** to ensure performance and interpretability:")

    # Chia cột để so sánh 2 phương pháp như trong báo cáo
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(
            """
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; height: 100%;">
                <h4 style="color: #1565c0; text-align: center;">🇬🇧 English Model</h4>
                <p style="text-align: center;"><b>Machine Learning</b></p>
                <hr>
                <ul>
                    <li><b>Algorithm:</b> Logistic Regression (sklearn).</li>
                    <li><b>Feature Extraction:</b> TF-IDF Vectorizer.</li>
                    <li><b>Why?</b> High speed, interpretability, and efficiency for text classification.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; height: 100%;">
                <h4 style="color: #e65100; text-align: center;">🇻🇳 Vietnamese Model</h4>
                <p style="text-align: center;"><b>Rule-Based (Heuristic)</b></p>
                <hr>
                <ul>
                    <li><b>Algorithm:</b> Dictionary-based matching.</li>
                    <li><b>Resources:</b> Predefined Sentiment Dictionaries (Positive/Negative keywords).</li>
                    <li><b>Why?</b> Effective for Vietnamese without requiring large labeled datasets.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Hiển thị Tech Stack bằng các Badge (Huy hiệu)
    st.write("")
    st.markdown("**🛠️ Tech Stack:**")
    
    # Dùng HTML để tạo các badge đẹp mắt
    st.markdown(
        """
        <style>
        .badge {
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 15px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }
        </style>
        <div>
            <span class="badge" style="background-color: #306998;">Python 🐍</span>
            <span class="badge" style="background-color: #ff4b4b;">Streamlit 🎈</span>
            <span class="badge" style="background-color: #F7931E;">Scikit-learn ⚙️</span>
            <span class="badge" style="background-color: #150458;">Pandas 🐼</span>
            <span class="badge" style="background-color: #4CAF50;">Joblib 📦</span>
        </div>
        """,
        unsafe_allow_html=True
    )

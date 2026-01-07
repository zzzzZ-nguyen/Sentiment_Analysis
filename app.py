import streamlit as st

def show():
    # ==========================
    # 🎨 CSS STYLING CHO TRANG HOME
    # ==========================
    st.markdown(
        """
        <style>
        /* Style cho các Box phương pháp */
        .method-box {
            padding: 20px;
            border-radius: 15px;
            height: 100%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .method-box:hover {
            transform: translateY(-5px);
        }
        
        /* Style cho Badge công nghệ */
        .tech-badge {
            display: inline-block;
            padding: 6px 12px;
            margin: 5px;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Chỉnh font cho tiêu đề */
        .section-title {
            color: #2b6f3e;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # 1. PROBLEM OVERVIEW (BỐI CẢNH)
    # ==========================================
    st.markdown("<h3 class='section-title'>1. Problem Overview</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            <div style="text-align: justify; font-size: 1.05rem; line-height: 1.6;">
            In the rapidly expanding digital economy, e-commerce platforms generate massive amounts of 
            <b>unstructured data</b> in the form of customer product reviews. 
            <br><br>
            For businesses, manually analyzing thousands of reviews to understand customer satisfaction is:
            <ul style="list-style-type: none; padding-left: 0;">
                <li>❌ <b>Time-consuming:</b> Impossible to scale with human effort alone.</li>
                <li>❌ <b>Expensive:</b> High operational costs for manual labeling.</li>
                <li>❌ <b>Prone to Error:</b> Subjective bias in human interpretation.</li>
            </ul>
            ⚠️ This creates a <i>"data-rich, information-poor"</i> scenario where valuable insights into product quality are lost.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2920/2920349.png", 
            caption="Information Overload Challenge",
            width=200
        )

    st.write("") # Spacer

    # ==========================================
    # 2. OBJECTIVES (MỤC TIÊU)
    # ==========================================
    st.markdown("<h3 class='section-title'>2. Objectives</h3>", unsafe_allow_html=True)
    
    st.success(
        "🎯 **Primary Goal:** Develop a lightweight, bilingual **Sentiment Analysis Application** using Python."
    )
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### ⚡ Automated")
        st.caption("Classify Positive, Neutral, Negative instantly.")
    with c2:
        st.markdown("#### 🌏 Bilingual")
        st.caption("Support both English (Global) & Vietnamese (Local).")
    with c3:
        st.markdown("#### ⏱️ Real-time")
        st.caption("Immediate inference via Web Interface.")
    with c4:
        st.markdown("#### 📈 Insights")
        st.caption("Support data-driven business decisions.")

    st.write("") # Spacer

    # ==========================================
    # 3. METHODOLOGY (PHƯƠNG PHÁP)
    # ==========================================
    st.markdown("<h3 class='section-title'>3. Technologies & Methodology</h3>", unsafe_allow_html=True)
    
    st.markdown("This system utilizes a **Hybrid Approach** to ensure performance and interpretability:")

    meth_col1, meth_col2 = st.columns(2)
    
    with meth_col1:
        st.markdown(
            """
            <div class="method-box" style="background-color: #e3f2fd; border: 1px solid #bbdefb;">
                <h4 style="color: #1565c0; text-align: center;">🇬🇧 English Model</h4>
                <p style="text-align: center; font-weight: bold; color: #0d47a1;">Machine Learning</p>
                <hr style="border-top: 1px solid #90caf9;">
                <ul>
                    <li><b>Algorithm:</b> Logistic Regression (sklearn).</li>
                    <li><b>Feature Extraction:</b> TF-IDF Vectorizer.</li>
                    <li><b>Performance:</b> 86% Accuracy.</li>
                    <li><b>Why?</b> High speed & Explainable AI.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with meth_col2:
        st.markdown(
            """
            <div class="method-box" style="background-color: #fff3e0; border: 1px solid #ffe0b2;">
                <h4 style="color: #e65100; text-align: center;">🇻🇳 Vietnamese Model</h4>
                <p style="text-align: center; font-weight: bold; color: #bf360c;">Rule-Based (Heuristic)</p>
                <hr style="border-top: 1px solid #ffcc80;">
                <ul>
                    <li><b>Algorithm:</b> Dictionary-based matching.</li>
                    <li><b>Resources:</b> Sentiment Keyword Dictionaries.</li>
                    <li><b>Logic:</b> <code>Score = Pos_count - Neg_count</code></li>
                    <li><b>Why?</b> Effective for limited datasets.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================================
    # 4. TECH STACK (CÔNG NGHỆ)
    # ==========================================
    st.write("")
    st.markdown("**🛠️ Tech Stack:**")
    
    st.markdown(
        """
        <div>
            <span class="tech-badge" style="background-color: #306998;">Python 🐍</span>
            <span class="tech-badge" style="background-color: #ff4b4b;">Streamlit 🎈</span>
            <span class="tech-badge" style="background-color: #F7931E;">Scikit-learn ⚙️</span>
            <span class="tech-badge" style="background-color: #150458;">Pandas 🐼</span>
            <span class="tech-badge" style="background-color: #4CAF50;">Joblib 📦</span>
            <span class="tech-badge" style="background-color: #2b6f3e;">WordCloud ☁️</span>
        </div>
        """,
        unsafe_allow_html=True
    )

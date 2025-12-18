import streamlit as st

def box(html):
    st.markdown(
        f"""
        <div style="
            background:#fff7cc;
            padding:20px;
            border-radius:10px;
            border:1px solid #e6d784;
            font-size:16px;
            line-height:1.6;">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

def show():
    st.markdown(
        "<h3 style='color:#2b6f3e;'>Topic 5 – Sentiment Analysis for Product Reviews</h3>",
        unsafe_allow_html=True
    )

    box("""
    <h4 style="color:#b30000;">1. Problem Overview</h4>
    The project develops an intelligent sentiment analysis system that automatically
    classifies product reviews into <b>Positive, Neutral, or Negative</b>
    to support decision-making for e-commerce businesses.
    """)

    box("""
    <h4 style="color:#b30000;">2. Objectives</h4>
    <ul>
        <li>Analyze customer opinions from product reviews.</li>
        <li>Support Vietnamese and English text.</li>
        <li>Visualize sentiment distribution.</li>
        <li>Provide real-time sentiment prediction.</li>
    </ul>
    """)

    box("""
    <h4 style="color:#b30000;">3. Technologies</h4>
    <ul>
        <li>Python, Streamlit</li>
        <li>Scikit-learn, TF-IDF</li>
        <li>Logistic Regression, SVM, XGBoost (optional)</li>
    </ul>
    """)

    st.write("---")
   

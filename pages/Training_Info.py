import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# Hàm show() bọc toàn bộ code
def show():
    # Load thư viện vẽ hình
    try:
        from wordcloud import WordCloud
        HAS_WORDCLOUD = True
    except ImportError: HAS_WORDCLOUD = False
    
    try:
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError: HAS_PLOTLY = False

    # Hàm đọc dữ liệu
    @st.cache_data
    def load_all_data():
        data_dir = "data"
        all_data = []
        if not os.path.exists(data_dir): return pd.DataFrame(), False
        
        # Files config
        files = {
            "Negative": "train_negative_tokenized.txt",
            "Neutral": "train_neutral_tokenized.txt", 
            "Positive": "train_positive_tokenized.txt"
        }
        
        for label, filename in files.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip(): all_data.append({"Content": line.strip(), "Label": label, "Type": "Train"})
        
        # Test file
        test_path = os.path.join(data_dir, "test_tokenized_ANS.txt")
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i in range(0, len(lines)-1, 2):
                    txt, lbl = lines[i].strip(), lines[i+1].strip()
                    label_map = {'NEG':'Negative', 'POS':'Positive', 'NEU':'Neutral'}
                    if txt: all_data.append({"Content": txt, "Label": label_map.get(lbl, 'Neutral'), "Type": "Test"})
                    
        return pd.DataFrame(all_data), True

    # --- UI CHÍNH ---
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("📊 Dữ Liệu Huấn Luyện")
    
    df, found = load_all_data()
    if not found:
        st.warning("Không tìm thấy dữ liệu trong thư mục `data/`")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng mẫu", len(df))
    c2.metric("Train Set", len(df[df['Type']=='Train']))
    c3.metric("Test Set", len(df[df['Type']=='Test']))
    
    st.divider()
    
    tab1, tab2 = st.tabs(["Biểu đồ", "Dữ liệu chi tiết"])
    
    with tab1:
        if HAS_PLOTLY:
            cnt = df['Label'].value_counts().reset_index()
            cnt.columns = ['Label', 'Count']
            fig = px.pie(cnt, values='Count', names='Label', hole=0.5, color='Label', 
                         color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df['Label'].value_counts())
            
    with tab2:
        st.dataframe(df, use_container_width=True, height=400)
    
    st.markdown('</div>', unsafe_allow_html=True)

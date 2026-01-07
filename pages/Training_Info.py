import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Hàm show() bọc toàn bộ code ---
def show():
    # --- Cấu hình thư viện vẽ hình ---
    try:
        from wordcloud import WordCloud
        HAS_WORDCLOUD = True
    except ImportError:
        HAS_WORDCLOUD = False

    try:
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    # (Lưu ý: KHÔNG CÓ st.set_page_config ở đây)

    # --- HÀM ĐỌC DỮ LIỆU ---
    @st.cache_data
    def load_all_data():
        data_dir = "data" 
        all_data = []
        
        # 1. ĐỌC TẬP TRAIN
        train_files = {
            "Negative": "train_negative_tokenized.txt",
            "Neutral": "train_neutral_tokenized.txt",
            "Positive": "train_positive_tokenized.txt"
        }
        
        for label, filename in train_files.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            all_data.append({"Content": line.strip(), "Label": label, "Type": "Train"})

        # 2. ĐỌC TẬP TEST
        test_path = os.path.join(data_dir, "test_tokenized_ANS.txt")
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i in range(0, len(lines) - 1, 2):
                    text = lines[i].strip()
                    label_code = lines[i+1].strip()
                    if label_code == 'NEG': label = "Negative"
                    elif label_code == 'POS': label = "Positive"
                    else: label = "Neutral"
                    if text:
                        all_data.append({"Content": text, "Label": label, "Type": "Test"})
        
        if not all_data: return pd.DataFrame(), False
        return pd.DataFrame(all_data), True

    # --- GIAO DIỆN CHÍNH ---
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("📊 Dữ Liệu & Huấn Luyện (Dashboard)")
    st.write("Tổng quan về bộ dữ liệu đã được làm sạch và sử dụng cho Model.")

    df, data_found = load_all_data()

    if not data_found:
        st.error("⚠️ Không tìm thấy dữ liệu trong thư mục `data/`.")
        st.stop()

    # --- METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Tổng mẫu", f"{len(df):,}")
    with col2: st.metric("Train", f"{len(df[df['Type']=='Train']):,}")
    with col3: st.metric("Test", f"{len(df[df['Type']=='Test']):,}")
    with col4: st.metric("Từ vựng (Ước tính)", f"{len(set(' '.join(df['Content'].astype(str)).split())):,}")

    st.divider()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Phân Bố", "☁️ Từ Khóa", "📋 Chi Tiết"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tỷ lệ cảm xúc")
            if HAS_PLOTLY:
                counts = df['Label'].value_counts().reset_index()
                counts.columns = ['Label', 'Count']
                fig = px.pie(counts, values='Count', names='Label', hole=0.5, color='Label',
                             color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df['Label'].value_counts())
        with c2:
            st.subheader("Train vs Test")
            if HAS_PLOTLY:
                type_counts = df.groupby(['Type', 'Label']).size().reset_index(name='Count')
                fig2 = px.bar(type_counts, x="Type", y="Count", color="Label", barmode="group",
                              color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("☁️ Word Cloud")
        if HAS_WORDCLOUD:
            sel = st.radio("Chọn nhãn:", ["Positive", "Negative", "Neutral"], horizontal=True)
            text = " ".join(df[df['Label'] == sel]['Content'].astype(str))
            if text:
                cmap = 'Greens' if sel == 'Positive' else 'Reds' if sel == 'Negative' else 'Oranges'
                wc = WordCloud(width=1000, height=400, background_color='white', colormap=cmap, max_words=100).generate(text)
                fig_wc, ax = plt.subplots(figsize=(12, 5))
                ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                st.pyplot(fig_wc)
            else: st.info("Không có dữ liệu.")
        else: st.warning("Chưa cài wordcloud.")

    with tab3:
        st.subheader("🔍 Dữ liệu thô")
        st.dataframe(df, use_container_width=True, height=500)
    
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Cấu hình thư viện vẽ hình (Để bên ngoài hàm show để load 1 lần) ---
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

# ==================================================
# 1. HÀM ĐỌC DỮ LIỆU (Helper Functions)
# ==================================================
@st.cache_data
def load_all_data():
    data_dir = "data" # Thư mục chứa file
    all_data = []
    
    # Kiểm tra thư mục data
    if not os.path.exists(data_dir):
        return pd.DataFrame(), False

    # 1. ĐỌC TẬP TRAIN (File txt thường)
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
                        all_data.append({
                            "Content": line.strip(), 
                            "Label": label, 
                            "Type": "Train"
                        })

    # 2. ĐỌC TẬP TEST (File đặc biệt: Dòng chẵn Text, Dòng lẻ Label)
    test_path = os.path.join(data_dir, "test_tokenized_ANS.txt")
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # File test: Text -> Label -> Text -> Label...
            for i in range(0, len(lines) - 1, 2):
                text = lines[i].strip()
                label_code = lines[i+1].strip()
                
                # Chuyển mã sang tên nhãn
                if label_code == 'NEG': label = "Negative"
                elif label_code == 'POS': label = "Positive"
                elif label_code == 'NEU': label = "Neutral"
                else: label = "Neutral"
                
                if text:
                    all_data.append({
                        "Content": text, 
                        "Label": label, 
                        "Type": "Test"
                    })
    
    # Nếu không đọc được dữ liệu nào
    if not all_data:
        return pd.DataFrame(), False

    return pd.DataFrame(all_data), True

# ==================================================
# 2. HÀM HIỂN THỊ CHÍNH (QUAN TRỌNG: PHẢI CÓ def show())
# ==================================================
def show():
    # Lưu ý: KHÔNG DÙNG st.set_page_config() Ở ĐÂY NỮA
    
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("📊 Dữ Liệu & Huấn Luyện (Dashboard)")
    st.write("Tổng quan về bộ dữ liệu đã được làm sạch và sử dụng cho Model.")

    # Load dữ liệu
    df, data_found = load_all_data()

    if not data_found:
        st.error("⚠️ Không tìm thấy dữ liệu trong thư mục `data/`. Vui lòng kiểm tra lại file.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # --- METRICS (Thống kê số lượng) ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng mẫu (Samples)", f"{len(df):,}")
    with col2:
        st.metric("Dữ liệu Train", f"{len(df[df['Type']=='Train']):,}")
    with col3:
        st.metric("Dữ liệu Test", f"{len(df[df['Type']=='Test']):,}")
    with col4:
        vocab_est = len(set(" ".join(df['Content'].astype(str)).split()))
        st.metric("Từ vựng (Ước tính)", f"{vocab_est:,}")

    st.divider()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Phân Bố (Charts)", "☁️ Từ Khóa (WordCloud)", "📋 Dữ Liệu Chi Tiết"])

    # TAB 1: CHARTS
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tỷ lệ cảm xúc (Sentiment)")
            counts = df['Label'].value_counts().reset_index()
            counts.columns = ['Label', 'Count']
            
            if HAS_PLOTLY:
                fig = px.pie(counts, values='Count', names='Label', hole=0.5,
                             color='Label',
                             color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df['Label'].value_counts())
                
        with c2:
            st.subheader("Số lượng Train vs Test")
            if HAS_PLOTLY:
                type_counts = df.groupby(['Type', 'Label']).size().reset_index(name='Count')
                fig2 = px.bar(type_counts, x="Type", y="Count", color="Label", barmode="group",
                              color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write(df['Type'].value_counts())

    # TAB 2: WORDCLOUD
    with tab2:
        st.subheader("☁️ Đám mây từ vựng (Word Cloud)")
        
        if HAS_WORDCLOUD:
            selected_sentiment = st.radio("Chọn loại cảm xúc để xem:", ["Positive", "Negative", "Neutral"], horizontal=True)
            
            # Lọc text
            subset = df[df['Label'] == selected_sentiment]
            text = " ".join(subset['Content'].astype(str))
            
            if text:
                # Tạo màu tùy chọn
                cmap = 'Greens' if selected_sentiment == 'Positive' else 'Reds' if selected_sentiment == 'Negative' else 'Oranges'
                
                wc = WordCloud(width=1000, height=400, background_color='white', colormap=cmap, max_words=100).generate(text)
                
                fig_wc, ax = plt.subplots(figsize=(12, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            else:
                st.info("Chưa có dữ liệu cho nhãn này.")
        else:
            st.warning("⚠️ Thư viện `wordcloud` chưa được cài đặt.")

    # TAB 3: DATA TABLE
    with tab3:
        st.subheader("🔍 Tra cứu dữ liệu thô")
        
        # Bộ lọc
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            type_filter = st.multiselect("Chọn tập dữ liệu:", ["Train", "Test"], default=["Train", "Test"])
        with filter_col2:
            label_filter = st.multiselect("Chọn nhãn:", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
        
        # Apply filter
        if type_filter and label_filter:
            df_show = df[df['Type'].isin(type_filter) & df['Label'].isin(label_filter)]
            st.dataframe(df_show, use_container_width=True, height=500)
        else:
            st.info("Vui lòng chọn ít nhất một bộ lọc.")
            
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import hàm load dữ liệu thông minh vừa sửa
from model_utils import load_training_data_for_app, load_lexicon_data

st.set_page_config(page_title="Data Info", page_icon="📊", layout="wide")

# CSS Style
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f0f2f6; border-radius: 10px; padding: 10px;
        border-left: 5px solid #2b6f3e;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Kho Dữ Liệu Tổng Hợp")
st.write("Tự động tổng hợp từ các file `.txt` và `.csv` (bao gồm `sentimentdataset.csv`) trong thư mục `data/`.")

# --- 1. LOAD DATA ---
with st.spinner("Đang quét dữ liệu..."):
    df = load_training_data_for_app()
    df_lexicon = load_lexicon_data()

if df.empty:
    st.error("⚠️ Không tìm thấy dữ liệu nào trong thư mục `data/`.")
    st.stop()

# --- 2. THỐNG KÊ TỔNG QUAN ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tổng số dòng dữ liệu", f"{len(df):,}")
with col2:
    n_files = df['Source'].nunique()
    st.metric("Số file nguồn", f"{n_files}", help=f"Gồm: {', '.join(df['Source'].unique())}")
with col3:
    n_labels = df['Label'].nunique()
    st.metric("Số loại nhãn", f"{n_labels}", help="Positive, Negative, Happy, Sad...")
with col4:
    if df_lexicon is not None:
        st.metric("Từ điển Lexicon", f"{len(df_lexicon):,}")
    else:
        st.metric("Từ điển Lexicon", "0")

st.divider()

# --- 3. BIỂU ĐỒ & PHÂN TÍCH ---
tab1, tab2, tab3 = st.tabs(["📈 Phân Bố Nhãn", "📋 Xem Dữ Liệu Chi Tiết", "☁️ WordCloud"])

# TAB 1: PHÂN BỐ
with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Tỷ lệ các loại cảm xúc")
        # Đếm số lượng mỗi nhãn
        label_counts = df['Label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        
        # Nếu có quá nhiều nhãn nhỏ (do file CSV mới có Happy, Joy...), gom lại
        if len(label_counts) > 10:
            st.info("💡 Dữ liệu có nhiều nhãn chi tiết (Happy, Joy...). Biểu đồ hiển thị Top 10 nhãn phổ biến nhất.")
            top_10 = label_counts.head(10)
            other_count = label_counts.iloc[10:]['Count'].sum()
            if other_count > 0:
                new_row = pd.DataFrame({'Label': ['Other'], 'Count': [other_count]})
                top_10 = pd.concat([top_10, new_row])
            fig = px.pie(top_10, values='Count', names='Label', hole=0.4)
        else:
            fig = px.pie(label_counts, values='Count', names='Label', hole=0.4)
            
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Nguồn dữ liệu")
        source_counts = df['Source'].value_counts()
        st.bar_chart(source_counts)

# TAB 2: DATA TABLE
with tab2:
    st.subheader("🔍 Tra cứu dữ liệu")
    
    # Filter
    all_sources = list(df['Source'].unique())
    selected_source = st.multiselect("Lọc theo file nguồn:", all_sources, default=all_sources)
    
    all_labels = list(df['Label'].unique())
    selected_label = st.multiselect("Lọc theo nhãn:", all_labels, default=all_labels[:3]) # Mặc định chọn 3 cái đầu
    
    # Apply filter
    df_show = df[df['Source'].isin(selected_source) & df['Label'].isin(selected_label)]
    
    st.dataframe(df_show, use_container_width=True, height=500)

# TAB 3: WORDCLOUD
with tab3:
    st.subheader("☁️ Đám mây từ vựng")
    
    # Chọn nhãn để vẽ
    top_labels = df['Label'].value_counts().head(5).index.tolist()
    chosen_label = st.selectbox("Chọn nhãn muốn xem:", top_labels)
    
    text_data = " ".join(df[df['Label'] == chosen_label]['Content'].astype(str))
    
    if text_data:
        try:
            wc = WordCloud(width=800, height=300, background_color='white', max_words=100).generate(text_data)
            fig_wc, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        except ValueError:
            st.warning("Không đủ từ vựng để tạo hình.")
    else:
        st.warning("Không có dữ liệu text cho nhãn này.")

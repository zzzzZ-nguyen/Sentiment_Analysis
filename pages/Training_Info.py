import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import load_training_data_for_app, load_lexicon_data, load_metadata_files

# ==========================================
# 1. ĐỊNH NGHĨA HÀM SHOW (QUAN TRỌNG)
# ==========================================
def show():
    # CSS Style
    st.markdown("""
    <style>
        div[data-testid="stMetric"] {
            background-color: #f0f2f6; border-radius: 10px; padding: 10px; border-left: 5px solid #2b6f3e;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Kho Dữ Liệu Tổng Hợp")

    # --- LOAD DATA ---
    with st.spinner("Đang quét dữ liệu..."):
        df = load_training_data_for_app()
        df_lexicon = load_lexicon_data()
        dict_metadata = load_metadata_files() # Load file metadata

    # --- THỐNG KÊ ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Dữ liệu Training", f"{len(df):,}")
    with col2: st.metric("Số file nguồn Train", f"{df['Source'].nunique()}")
    with col3: st.metric("Số file Metadata", f"{len(dict_metadata)}")
    with col4: st.metric("Từ điển Lexicon", f"{len(df_lexicon):,}" if df_lexicon is not None else "0")

    st.divider()

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Phân Bố Nhãn", "📋 Dữ Liệu Training", "☁️ WordCloud", "📂 Metadata (Mới)"])

    # TAB 1: CHART
    with tab1:
        if not df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Tỷ lệ cảm xúc")
                label_counts = df['Label'].value_counts().reset_index()
                label_counts.columns = ['Label', 'Count']
                if len(label_counts) > 10:
                    top_10 = label_counts.head(10)
                    other = pd.DataFrame({'Label': ['Other'], 'Count': [label_counts.iloc[10:]['Count'].sum()]})
                    label_counts = pd.concat([top_10, other])
                st.plotly_chart(px.pie(label_counts, values='Count', names='Label', hole=0.4), use_container_width=True)
            with c2:
                st.subheader("Nguồn dữ liệu")
                st.bar_chart(df['Source'].value_counts())
        else: st.info("Chưa có dữ liệu Training.")

    # TAB 2: TRAINING DATA TABLE
    with tab2:
        if not df.empty:
            st.subheader("🔍 Tra cứu dữ liệu Training")
            src = st.multiselect("Lọc file:", df['Source'].unique(), default=df['Source'].unique())
            lbl = st.multiselect("Lọc nhãn:", df['Label'].unique(), default=df['Label'].unique()[:3])
            
            # Filter logic
            if src and lbl:
                st.dataframe(df[df['Source'].isin(src) & df['Label'].isin(lbl)], use_container_width=True)
            else:
                st.warning("Vui lòng chọn ít nhất 1 File và 1 Nhãn.")
        else: st.warning("Trống.")

    # TAB 3: WORDCLOUD
    with tab3:
        if not df.empty:
            lbl = st.selectbox("Chọn nhãn:", df['Label'].value_counts().head(5).index.tolist())
            txt = " ".join(df[df['Label'] == lbl]['Content'].astype(str))
            if txt:
                try:
                    wc = WordCloud(width=800, height=300, background_color='white').generate(txt)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    st.pyplot(fig)
                except: st.warning("Không đủ dữ liệu.")
        else: st.warning("Trống.")

    # TAB 4: METADATA (HIỂN THỊ FILE MỚI)
    with tab4:
        st.subheader("📂 Dữ Liệu Tham Khảo (Metadata)")
        st.info("Đây là các file chứa thông tin bổ trợ (như mã ngôn ngữ, danh sách ID...), KHÔNG dùng để huấn luyện model.")
        
        if dict_metadata:
            chosen_meta = st.selectbox("Chọn file để xem:", list(dict_metadata.keys()))
            st.write(f"Đang xem: **{chosen_meta}**")
            st.dataframe(dict_metadata[chosen_meta], use_container_width=True)
        else:
            st.warning("Không tìm thấy file nào có tên chứa 'metadata' hoặc 'Metadata' trong thư mục data/.")

# ==========================================
# 2. KHỐI CHẠY ĐỘC LẬP (OPTIONAL)
# ==========================================
if __name__ == "__main__":
    # Chỉ set page config khi chạy file này trực tiếp (không qua app.py)
    st.set_page_config(page_title="Data Info", page_icon="📊", layout="wide")
    show()

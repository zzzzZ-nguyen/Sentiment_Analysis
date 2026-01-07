import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

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

# ==================================================
# 1. CẤU HÌNH GIAO DIỆN
# ==================================================
st.set_page_config(page_title="Data & Training Info", page_icon="📊", layout="wide")

# CSS Vintage Style
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #F0EBD6;
        background-image: repeating-linear-gradient(45deg, #F0EBD6 0, #F0EBD6 2px, #E8E4CC 2px, #E8E4CC 4px);
    }
    h1, h2, h3 { color: #2b6f3e !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px; border-radius: 8px;
        border-left: 5px solid #2b6f3e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 2. HÀM ĐỌC DỮ LIỆU
# ==================================================
@st.cache_data
def load_training_data():
    """Đọc dữ liệu Train/Test"""
    data_dir = "data"
    all_data = []
    
    # 1. ĐỌC TẬP TRAIN
    train_files = {
        "Negative": "train_negative_tokenized.txt",
        "Neutral": "train_neutral_tokenized.txt",
        "Positive": "train_positive_tokenized.txt"
    }
    
    if not os.path.exists(data_dir): return pd.DataFrame(), False

    for label, filename in train_files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        all_data.append({"Content": line.strip(), "Label": label, "Type": "Train"})

    # 2. ĐỌC TẬP TEST (Nếu có)
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
    
    return pd.DataFrame(all_data), True

@st.cache_data
def load_lexicon_data():
    """Đọc dữ liệu Từ điển (Lexicon) có xử lý lỗi dòng hỏng"""
    file_path = "data/vietnamese_lexicon.txt"
    if not os.path.exists(file_path): return None
    
    lexicon_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # Bỏ qua dòng trống hoặc comment
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            
            # Cấu trúc mong đợi: [Loại từ, ID, PosScore, NegScore, Word#ID, Definition...]
            if len(parts) >= 5:
                try:
                    # Thêm try-except để nếu dòng nào số liệu sai thì bỏ qua luôn
                    pos_score = float(parts[2])
                    neg_score = float(parts[3])
                    
                    word = parts[4].split('#')[0].replace('_', ' ') 
                    definition = " ".join(parts[5:]).strip('"')
                    
                    lexicon_data.append({
                        "Từ vựng": word,
                        "Loại từ": parts[0],
                        "Điểm Tích cực": pos_score,
                        "Điểm Tiêu cực": neg_score,
                        "Định nghĩa": definition
                    })
                except ValueError:
                    # Nếu dòng này không phải số (ví dụ dòng header), bỏ qua nó
                    continue
                    
    if not lexicon_data:
        return None
        
    return pd.DataFrame(lexicon_data)

# ==================================================
# 3. GIAO DIỆN CHÍNH
# ==================================================
st.title("📊 Dashboard Dữ liệu & Từ điển")
st.write("Quản lý dữ liệu huấn luyện và từ điển cảm xúc.")

df_train, found_train = load_training_data()
df_lexicon = load_lexicon_data()

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mẫu Train/Test", f"{len(df_train):,}" if found_train else "0")
with col2:
    st.metric("Từ trong từ điển", f"{len(df_lexicon):,}" if df_lexicon is not None else "0")
with col3:
    if found_train:
        pos_cnt = len(df_train[df_train['Label']=='Positive'])
        st.metric("Mẫu Tích cực", f"{pos_cnt:,}")
    else: st.metric("Mẫu Tích cực", "0")
with col4:
    if df_lexicon is not None:
        avg_pos = df_lexicon['Điểm Tích cực'].mean()
        st.metric("Pos Score TB", f"{avg_pos:.2f}")
    else: st.metric("Pos Score TB", "0")

st.divider()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📚 Từ Điển Cảm Xúc", "📈 Phân Bố", "☁️ WordCloud", "📋 Dữ Liệu Train"])

# TAB 1: TỪ ĐIỂN (NEW)
with tab1:
    st.subheader("📚 Từ điển SentiWordNet (Vietnamese)")
    if df_lexicon is not None:
        st.dataframe(
            df_lexicon, 
            column_config={
                "Điểm Tích cực": st.column_config.ProgressColumn("Positive Score", format="%.2f", min_value=0, max_value=1, help="Điểm càng cao càng tích cực"),
                "Điểm Tiêu cực": st.column_config.ProgressColumn("Negative Score", format="%.2f", min_value=0, max_value=1, help="Điểm càng cao càng tiêu cực"),
            },
            use_container_width=True,
            height=500
        )
    else:
        st.warning("⚠️ Chưa tìm thấy file `data/vietnamese_lexicon.txt`.")
        st.info("Hãy tạo file này và dán dữ liệu từ điển vào.")

# TAB 2: CHARTS
with tab2:
    if found_train and not df_train.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tỷ lệ nhãn")
            counts = df_train['Label'].value_counts().reset_index()
            counts.columns = ['Label', 'Count']
            if HAS_PLOTLY:
                fig = px.pie(counts, values='Count', names='Label', hole=0.5, color='Label',
                             color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df_train['Label'].value_counts())
        with c2:
            st.subheader("Train vs Test")
            st.bar_chart(df_train['Type'].value_counts())
    else:
        st.info("Chưa có dữ liệu Train/Test để vẽ biểu đồ.")

# TAB 3: WORDCLOUD
with tab3:
    st.subheader("☁️ Đám mây từ vựng")
    if found_train and HAS_WORDCLOUD:
        selected_sentiment = st.radio("Chọn nhãn:", ["Positive", "Negative"], horizontal=True)
        subset = df_train[df_train['Label'] == selected_sentiment]
        text = " ".join(subset['Content'].astype(str))
        if text:
            wc = WordCloud(width=800, height=300, background_color='white').generate(text)
            fig_wc, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        else:
            st.warning("Không đủ dữ liệu để tạo WordCloud.")

# TAB 4: DATA TABLE
with tab4:
    st.subheader("🔍 Dữ liệu Huấn luyện thô")
    if found_train:
        st.dataframe(df_train, use_container_width=True)
    else:
        st.warning("Không có dữ liệu.")


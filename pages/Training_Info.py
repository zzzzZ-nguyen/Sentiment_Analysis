import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==================================================
# 1. CẤU HÌNH TRANG (Bắt buộc đầu tiên)
# ==================================================
st.set_page_config(page_title="Training Info", page_icon="⚙️", layout="wide")

# ==================================================
# 2. CSS GIAO DIỆN
# ==================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F0EBD6;
    background-image: repeating-linear-gradient(45deg, #F0EBD6, #F0EBD6 20px, #BBDEA4 20px, #BBDEA4 40px);
}
div[data-testid="stTable"], div[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
    padding: 10px; border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
h1, h2, h3 { color: #2b6f3e; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. DỮ LIỆU RAW (Bạn dán dữ liệu vào đây)
# ==================================================
# Đây là dữ liệu dạng SentiWordNet bạn cung cấp
RAW_TEXT_DATA = """
a   001937946   0.125   0.5 ẩm_ướt#1    ẩm, do thấm nhiều nước hoặc có chứa nhiều hơi nước; "nền nhà ẩm ướt"
a   001937947   0.25    0.5 ân_hận#1    băn khoăn, day dứt và tự trách mình đã để xảy ra việc không hay ; "hắn ân hận về những gì đã làm với cô ấy"
n   001937948   0.5     0       ân_nghĩa#1  tình nghĩa thắm thiết, gắn bó do có chịu ơn sâu với nhau ; "con cái có ân nghĩa với cha mẹ"
a   001937949   0.5     0.25    ẩn_nấp#1    giấu mình ở nơi kín đáo hoặc nơi có vật che chở; "xuống hầm ẩn nấp tránh nạn" 
n   001937950   0.5     0.25    ẩn_ý#1  ý kín đáo bên trong, vốn là cái chính muốn nói, nhưng không nói rõ, chỉ để ngầm hiểu ; "lời nói có ẩn ý"
v   001937951   0.25    0.5 ấp_úng#1    từ gợi tả cách nói không nên lời hoặc nói không gãy gọn, không rành mạch vì lúng túng ; "nó trả lòi ấp úng"
a   001937952   0       0.5 bạc_đãi#1   đối xử rẻ rúng (với cái lẽ ra phải được coi trọng); "hắn bạc đãi với người làm thuê" 
v   001937953   0.25    0.5 bãi_nhiệm#1 bãi bỏ chức vụ (thường là quan trọng) trong bộ máy nhà nước (của người nào đó) ; "thủ tướng bị bãi nhiệm"
"""

# ==================================================
# 4. HÀM XỬ LÝ (PARSER & MODEL)
# ==================================================

def parse_sentiwordnet_data(raw_text):
    """Chuyển đổi text thô thành DataFrame"""
    rows = []
    lines = raw_text.strip().split('\n')
    
    for line in lines:
        parts = line.split() # Tách bằng khoảng trắng
        if len(parts) < 5: continue
        
        # Cấu trúc: [Type] [ID] [PosScore] [NegScore] [Word#Sense] [Definition...]
        try:
            pos_score = float(parts[2])
            neg_score = float(parts[3])
            word = parts[4].split('#')[0].replace('_', ' ') # Lấy từ, bỏ #1, bỏ gạch dưới
            definition = " ".join(parts[5:]) # Nối lại phần định nghĩa
            
            # Quy luật gán nhãn dựa trên điểm số
            if pos_score > neg_score:
                label = "positive"
            elif neg_score > pos_score:
                label = "negative"
            else:
                label = "neutral"
                
            rows.append({
                "Word": word,
                "Pos_Score": pos_score,
                "Neg_Score": neg_score,
                "Label": label,
                "Definition": definition
            })
        except:
            continue
            
    return pd.DataFrame(rows)

@st.cache_resource
def load_model_objects():
    paths = [os.path.join("models", "model_en.pkl"), os.path.join("..", "models", "model_en.pkl")]
    for p in paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                vec_path = p.replace("model_en.pkl", "vectorizer_en.pkl")
                vectorizer = joblib.load(vec_path)
                return model, vectorizer
            except: pass
    return None, None

# ==================================================
# 5. GIAO DIỆN CHÍNH
# ==================================================

st.markdown("<h2 style='text-align: center;'>⚙️ Training Information & Data Analysis</h2>", unsafe_allow_html=True)
st.info("Trang này hiển thị dữ liệu gốc (SentiWordNet Vietnamese) và hiệu suất của mô hình.")
st.write("---")

# --- PHẦN 1: DỮ LIỆU ĐẦU VÀO (RAW DATASET) ---
st.subheader("1️⃣ Raw Dataset (Vietnamese SentiWordNet)")
st.write("Dữ liệu gốc bao gồm điểm số Tích cực/Tiêu cực cho từng từ vựng.")

# Gọi hàm parser để xử lý dữ liệu bạn dán vào
df_raw = parse_sentiwordnet_data(RAW_TEXT_DATA)

# Hiển thị bảng dữ liệu
st.dataframe(df_raw, use_container_width=True)

# Hiển thị biểu đồ phân bố nhãn từ dữ liệu raw
col1, col2 = st.columns([2, 1])
with col1:
    st.caption("Mẫu dữ liệu sau khi làm sạch:")
    st.code(RAW_TEXT_DATA.split('\n')[1], language='text') # Hiện 1 dòng mẫu
with col2:
    st.caption("Thống kê nhãn:")
    st.bar_chart(df_raw['Label'].value_counts())

st.write("---")

# --- PHẦN 2: KẾT QUẢ TRAINING (MODEL METRICS) ---
st.subheader("2️⃣ Training Performance Metrics")

model, vectorizer = load_model_objects()

if model and vectorizer:
    # Nếu có model thật, ta thử dự đoán lại trên chính các từ vựng này
    # (Lưu ý: Model huấn luyện câu, dự đoán từ đơn có thể không chính xác tuyệt đối, đây chỉ là demo)
    X_test = vectorizer.transform(df_raw["Word"])
    y_pred = model.predict(X_test)
    y_true = df_raw["Label"] # Nhãn tính từ điểm số
    
    # Tính toán chỉ số
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Hiển thị
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy (Độ chính xác)", f"{acc*100:.1f}%", delta="Model vs Dictionary")
    m2.metric("F1-Score", f"{f1:.4f}")
    m3.metric("Vocabulary Size", f"{len(df_raw)} words")
    
    # Confusion Matrix
    st.markdown("##### Confusion Matrix")
    classes = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    st.dataframe(cm_df.style.background_gradient(cmap="Greens"))
    
else:
    # --- CHẾ ĐỘ GIẢ LẬP (KHI CHƯA CÓ FILE MODEL) ---
    st.warning("⚠️ Đang hiển thị dữ liệu giả lập (Chưa load được file model thực tế).")
    
    # Số liệu giả
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "87.5%", "+1.2%")
    m2.metric("F1-Score", "0.8542")
    m3.metric("Total Samples", "10,500")
    
    # Bảng nhầm lẫn giả
    dummy_cm = pd.DataFrame(
        [[50, 5, 2], [3, 40, 4], [1, 2, 60]], 
        index=["Negative", "Neutral", "Positive"], 
        columns=["Negative", "Neutral", "Positive"]
    )
    st.table(dummy_cm)

st.write("---")

# --- PHẦN 3: CHI TIẾT TỪ ĐIỂN ---
st.subheader("3️⃣ Từ điển & Điểm số chi tiết")
st.markdown("Mô hình phân tích cảm xúc dựa trên việc học các trọng số từ vựng sau:")

# Tô màu cho bảng dựa trên điểm số
def highlight_sentiment(row):
    if row.Pos_Score > row.Neg_Score:
        return ['background-color: #d4edda; color: black'] * len(row) # Xanh lá
    elif row.Neg_Score > row.Pos_Score:
        return ['background-color: #f8d7da; color: black'] * len(row) # Đỏ nhạt
    else:
        return ['background-color: #fff3cd; color: black'] * len(row) # Vàng

st.dataframe(df_raw[["Word", "Pos_Score", "Neg_Score", "Definition"]].style.apply(highlight_sentiment, axis=1), use_container_width=True)

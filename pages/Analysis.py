import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ==================================================
# ⚙️ CẤU HÌNH TRANG
# ==================================================
st.set_page_config(page_title="Smart Sentiment Analysis", page_icon="🧠", layout="wide")

# File lưu trữ dữ liệu lịch sử
HISTORY_FILE = "data/history_log.csv"

# Tạo thư mục data nếu chưa có
if not os.path.exists("data"):
    os.makedirs("data")

# ==================================================
# 📦 HÀM HỖ TRỢ (LOAD MODEL & STORAGE)
# ==================================================
@st.cache_resource
def load_model():
    # Sửa đường dẫn phù hợp với máy của bạn
    paths = [os.path.join("models", "model_en.pkl"), os.path.join("..", "models", "model_en.pkl")]
    for p in paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                vec_path = p.replace("model_en.pkl", "vectorizer_en.pkl")
                vectorizer = joblib.load(vec_path)
                return model, vectorizer
            except:
                continue
    return None, None

def save_to_history(text, predicted_label, user_correction=None):
    """Lưu dữ liệu vào file CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Nhãn cuối cùng (nếu user sửa thì lấy user sửa, không thì lấy máy đoán)
    final_label = user_correction if user_correction else predicted_label
    
    new_data = pd.DataFrame({
        "Timestamp": [timestamp],
        "Text": [text],
        "Predicted": [predicted_label],
        "Corrected_Label": [final_label], # Nhãn chuẩn để train lại sau này
        "Is_Correction": [user_correction is not None] # Đánh dấu dòng nào do người dùng sửa
    })

    if not os.path.exists(HISTORY_FILE):
        new_data.to_csv(HISTORY_FILE, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Timestamp", "Text", "Predicted", "Corrected_Label", "Is_Correction"])

# ==================================================
# 🖥️ GIAO DIỆN CHÍNH
# ==================================================
st.markdown("<h2 style='color:#2b6f3e;'>🧠 Smart Sentiment Analysis</h2>", unsafe_allow_html=True)
st.write("Hệ thống phân tích cảm xúc có khả năng ghi nhớ và thu thập dữ liệu huấn luyện.")

model, vectorizer = load_model()

if not model:
    st.error("⚠️ Không tìm thấy Model. Vui lòng kiểm tra thư mục 'models/'.")
    st.stop()

col1, col2 = st.columns([2, 1])

# --- CỘT TRÁI: NHẬP LIỆU & DỰ ĐOÁN ---
with col1:
    st.subheader("1. Phân Tích")
    user_input = st.text_area("Nhập nội dung đánh giá (Review):", height=150, placeholder="Ví dụ: Sản phẩm này dùng rất thích...")
    
    # Biến session state để giữ kết quả sau khi reload
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    if st.button("🚀 Phân Tích Ngay", type="primary"):
        if user_input.strip():
            # Xử lý dự đoán
            text_vec = vectorizer.transform([user_input.lower()])
            pred = model.predict(text_vec)[0]
            prob = model.predict_proba(text_vec).max()
            
            # Lưu vào session để hiển thị
            st.session_state.prediction_result = {
                "text": user_input,
                "label": pred,
                "score": prob
            }
            # Tự động lưu log ban đầu
            save_to_history(user_input, pred) 
        else:
            st.warning("Vui lòng nhập nội dung!")

    # HIỂN THỊ KẾT QUẢ & SỬA LỖI
    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        
        st.divider()
        st.markdown("### Kết quả:")
        
        # Hiển thị màu sắc dựa trên kết quả
        color_map = {"positive": "success", "negative": "error", "neutral": "warning"}
        msg_func = getattr(st, color_map.get(res['label'], "info"))
        
        msg_func(f"Dự đoán: **{res['label'].upper()}** (Độ tin cậy: {res['score']:.2%})")
        
        # --- PHẦN FEEDBACK (QUAN TRỌNG) ---
        with st.expander("🛠️ Báo cáo sai / Sửa nhãn đúng"):
            st.write("Nếu máy dự đoán sai, hãy chọn nhãn đúng bên dưới để giúp máy học tốt hơn:")
            correct_label = st.radio("Nhãn chính xác là:", model.classes_, horizontal=True)
            
            if st.button("💾 Cập nhật dữ liệu"):
                if correct_label != res['label']:
                    save_to_history(res['text'], res['label'], user_correction=correct_label)
                    st.success("Đã lưu phản hồi! Dữ liệu này sẽ được dùng để train lại model.")
                    # Xóa session để reset
                    del st.session_state.prediction_result
                    st.rerun()
                else:
                    st.info("Nhãn bạn chọn trùng với dự đoán. Không cần cập nhật.")

# --- CỘT PHẢI: LỊCH SỬ & DỮ LIỆU ---
with col2:
    st.subheader("2. Lịch sử & Dữ liệu")
    
    # Load và hiển thị lịch sử
    history_df = load_history()
    
    if not history_df.empty:
        # Đảo ngược để thấy cái mới nhất lên đầu
        display_df = history_df.iloc[::-1].head(10)
        
        # Hiển thị dạng bảng nhỏ
        st.dataframe(
            display_df[["Text", "Corrected_Label"]], 
            hide_index=True,
            use_container_width=True
        )
        
        st.caption(f"Tổng số dữ liệu đã lưu: {len(history_df)} dòng")
        
        # Nút tải dữ liệu về
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải trọn bộ Dataset (.csv)",
            data=csv_data,
            file_name="sentiment_history_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Chưa có lịch sử phân tích nào.")

# CSS làm đẹp
st.markdown("""
<style>
div.stButton > button {width: 100%; border-radius: 5px;}
[data-testid="stSidebar"] {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

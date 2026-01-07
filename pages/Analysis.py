import streamlit as st
import sys
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Analysis PyTorch", page_icon="🧠", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
div.stButton > button {
    background-color: #2b6f3e; color: white; border-radius: 5px; width: 100%; font-weight: bold;
}
.stTextArea textarea { background-color: #f0f2f6; color: #333; }
</style>
""", unsafe_allow_html=True)

# --- XỬ LÝ IMPORT TỪ THƯ MỤC GỐC ---
# Lấy đường dẫn thư mục hiện tại (pages/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn thư mục cha (thư mục gốc chứa model_utils.py)
parent_dir = os.path.dirname(current_dir)
# Thêm vào sys.path để Python tìm thấy file
sys.path.append(parent_dir)

try:
    from model_utils import load_model_resources, predict
    HAS_UTILS = True
except ImportError as e:
    HAS_UTILS = False
    st.error(f"❌ Lỗi Import: Không tìm thấy file `model_utils.py`. Chi tiết: {e}")
    st.info("💡 Giải pháp: Hãy tạo file `model_utils.py` ở thư mục gốc (cùng chỗ với app.py).")
    st.stop() # Dừng chương trình tại đây nếu lỗi

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h2 style='color:#2b6f3e;'>🧠 Deep Learning Sentiment Analysis</h2>", unsafe_allow_html=True)
st.write("Phân tích cảm xúc sử dụng mô hình LSTM (PyTorch).")

# 1. Load Model
vocab, model = load_model_resources()

if model is None:
    st.warning("⚠️ Chưa tìm thấy Model hợp lệ.")
    st.markdown("""
    **Nguyên nhân:**
    1. Bạn chưa chạy huấn luyện ở trang **Train PyTorch**.
    2. File `models/sentiment_model.pth` hoặc `models/vocab.pkl` bị thiếu.
    
    👉 **Khắc phục:** Vui lòng sang trang **Train PyTorch** và bấm nút **Train Model**.
    """)
    st.stop()

# 2. Giao diện Phân tích
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Nhập nội dung")
    user_input = st.text_area("Review của khách hàng:", height=150, placeholder="Ví dụ: Hàng dùng rất tốt, giao hàng nhanh...")
    
    if st.button("🚀 Phân tích ngay"):
        if user_input.strip():
            with st.spinner("Đang tính toán..."):
                # Gọi hàm dự đoán
                score = predict(user_input, vocab, model)
            
            st.write("---")
            st.markdown("### 🎯 Kết quả phân tích")
            
            # Hiển thị kết quả với thanh tiến trình
            st.progress(score)
            
            if score >= 0.6:
                st.success(f"**TÍCH CỰC (POSITIVE) 😊**\n\nĐộ tin cậy: {score:.2%}")
                st.balloons()
            elif score <= 0.4:
                st.error(f"**TIÊU CỰC (NEGATIVE) 😡**\n\nĐộ tin cậy: {(1-score):.2%}")
            else:
                st.warning(f"**TRUNG TÍNH (NEUTRAL) 😐**\n\nĐiểm số: {score:.2f}")
        else:
            st.warning("Vui lòng nhập nội dung trước khi bấm nút.")

with col2:
    st.markdown("### ℹ️ Ví dụ mẫu")
    st.info("**Tích cực:**\n- Sản phẩm tuyệt vời.\n- Shop tư vấn nhiệt tình.")
    st.error("**Tiêu cực:**\n- Hàng lởm, đừng mua.\n- Vừa nhận đã hỏng.")
    st.warning("**Trung tính:**\n- Dùng cũng tạm.\n- Không có gì đặc sắc.")

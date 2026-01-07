import streamlit as st
import sys
import os

# Thêm đường dẫn thư mục gốc để import được model_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model_utils import load_model_resources, predict
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

def show():
    # CSS Styling
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #2b6f3e; color: white; border-radius: 5px; width: 100%; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#2b6f3e;'>🧠 Deep Learning Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    if not HAS_UTILS:
        st.error("⚠️ Không tìm thấy file `model_utils.py` ở thư mục gốc. Vui lòng tạo file này trước.")
        return

    # Load Model từ file Utils
    vocab, model = load_model_resources()

    if model is None:
        st.warning("⚠️ Chưa tìm thấy Model đã train.")
        st.info("👉 Vui lòng vào trang **Train PyTorch**, tạo dữ liệu và bấm nút 'Train' để tạo model trước.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Input Review")
        user_input = st.text_area("Nhập nội dung đánh giá:", height=150, placeholder="Ví dụ: Sản phẩm dùng rất tốt, tôi rất thích...")
        
        if st.button("🚀 Analyze Sentiment"):
            if user_input.strip():
                with st.spinner("Đang phân tích..."):
                    # Gọi hàm dự đoán từ model_utils
                    score = predict(user_input, vocab, model)
                
                st.write("---")
                st.markdown("### 🎯 Result")
                
                if score >= 0.6:
                    st.success(f"**POSITIVE (Tích cực)**\n\nĐộ tin cậy: {score:.2%}")
                    st.balloons()
                elif score <= 0.4:
                    st.error(f"**NEGATIVE (Tiêu cực)**\n\nĐộ tin cậy: {(1-score):.2%}")
                else:
                    st.warning(f"**NEUTRAL (Trung tính)**\n\nĐiểm số: {score:.2f}")
            else:
                st.warning("Vui lòng nhập nội dung văn bản.")

    with col2:
        st.markdown("### ℹ️ Examples")
        st.info("**Positive:**\n- Sản phẩm dùng rất tốt.\n- Giao hàng nhanh.")
        st.error("**Negative:**\n- Hàng kém chất lượng.\n- Thái độ phục vụ tồi.")

    st.write("---")

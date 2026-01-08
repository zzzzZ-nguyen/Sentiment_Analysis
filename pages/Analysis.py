import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import random

ef show():
    # ==================================================
    # PASTE ALL YOUR ORIGINAL CODE FROM Analysis.py HERE
    # ==================================================
    
    st.header("🔍 Sentiment Analysis Prediction")
    
    # Example content:
    user_input = st.text_area("Enter product review:")
    
    if st.button("Predict"):
        if user_input:
            # Your prediction logic here
            st.success("Prediction result goes here")
        else:
            st.warning("Please enter some text.")

# Make sure this line is NOT here or is inside a name check if you run it standalone
# show()  <-- DELETE THIS if it exists at the bottom indentation level
# Import từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import load_model_resources, predict_debug, load_training_data_for_app

# --- CONFIG PAGE ---
st.set_page_config(page_title="Deep Analysis", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px; }
    .result-box {
        padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;
        font-weight: bold; font-size: 20px; color: white;
    }
    .pos { background-color: #28a745; }
    .neg { background-color: #dc3545; }
    .neu { background-color: #ffc107; color: black !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Phân Tích Cảm Xúc Chuyên Sâu")
st.write("Test model với dữ liệu nhập tay hoặc lấy ngẫu nhiên từ tập Training.")

# --- 1. LOAD TÀI NGUYÊN ---
vocab, model = load_model_resources()

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

col_main, col_sidebar = st.columns([2, 1])

# --- 2. CỘT PHẢI: CÔNG CỤ DATA ---
with col_sidebar:
    st.markdown("### 🎲 Dữ liệu mẫu")
    st.info("Lấy ngẫu nhiên 1 câu trong dữ liệu `Training_Info` để kiểm tra độ học của máy.")
    
    if st.button("🔄 Lấy mẫu ngẫu nhiên", use_container_width=True):
        df = load_training_data_for_app() # Gọi hàm từ model_utils
        if not df.empty:
            sample = df.sample(1).iloc[0]
            st.session_state['input_text'] = sample['Content']
            # Lưu nhãn gốc để đối chiếu
            st.session_state['true_label'] = sample['Label'] 
            st.toast(f"Đã lấy mẫu: {sample['Label']}", icon="✅")
        else:
            st.error("Không tìm thấy dữ liệu trong folder `data/`")

    # Hiển thị nhãn gốc nếu có
    if 'true_label' in st.session_state and st.session_state['input_text']:
        st.caption(f"🏷️ Nhãn gốc trong data: **{st.session_state['true_label']}**")

# --- 3. CỘT TRÁI: PHÂN TÍCH ---
with col_main:
    # Text Area nhận giá trị từ Session State
    user_input = st.text_area("Nhập nội dung review:", 
                              value=st.session_state['input_text'], 
                              height=150,
                              placeholder="Ví dụ: Sản phẩm dùng rất chán, phí tiền...")
    
    if st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True):
        if not model:
            st.error("⚠️ Chưa có Model! Vui lòng qua trang **Train PyTorch** huấn luyện trước.")
        elif not user_input.strip():
            st.warning("Vui lòng nhập nội dung.")
        else:
            # --- XỬ LÝ DỰ ĐOÁN ---
            score, words, tokens = predict_debug(user_input, vocab, model)
            
            # --- HIỂN THỊ KẾT QUẢ ---
            st.divider()
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("#### Kết quả dự đoán:")
                if score >= 0.6:
                    st.markdown(f'<div class="result-box pos">TÍCH CỰC<br>{score:.2%}</div>', unsafe_allow_html=True)
                elif score <= 0.4:
                    st.markdown(f'<div class="result-box neg">TIÊU CỰC<br>{(1-score):.2%}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box neu">TRUNG TÍNH<br>{score:.2f}</div>', unsafe_allow_html=True)
            
            with c2:
                st.markdown("#### Độ tin cậy:")
                st.progress(score)
                if score > 0.5:
                    st.caption("Máy nghiêng về phía Tích cực.")
                else:
                    st.caption("Máy nghiêng về phía Tiêu cực.")

            # --- DEBUG INFO (QUAN TRỌNG) ---
            with st.expander("🔍 Soi kính lúp (Tại sao máy đoán vậy?)", expanded=True):
                st.write("**1. Máy đọc (Tokenization):**")
                
                # Tạo HTML để highlight từ lạ
                html_tokens = []
                unk_count = 0
                for w, idx in zip(words, tokens):
                    if idx == 0: # 0 là UNK (Unknown)
                        html_tokens.append(f'<span style="background-color:#ffcccc; padding:2px; border-radius:3px; color:red" title="Từ lạ (Không có trong Training)">{w} (?)</span>')
                        unk_count += 1
                    else:
                        html_tokens.append(f'<span style="background-color:#e6ffe6; padding:2px; border-radius:3px;">{w}</span>')
                
                st.markdown(" ".join(html_tokens), unsafe_allow_html=True)
                
                st.write("---")
                st.write(f"**Thống kê:** Tổng {len(words)} từ. Có **{unk_count}** từ lạ (UNK).")
                if unk_count > len(words) * 0.3:
                    st.warning("⚠️ **Cảnh báo:** Câu này chứa nhiều từ mà máy chưa từng học. Kết quả có thể không chính xác.")
                    st.info("💡 **Gợi ý:** Hãy thêm các từ này vào dữ liệu Train và huấn luyện lại.")

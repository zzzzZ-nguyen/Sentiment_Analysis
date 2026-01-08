import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# --- IMPORT UTILS (Xử lý đường dẫn để tìm file model_utils.py ở thư mục gốc) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model_utils import load_model_resources, predict_debug, load_training_data_for_app
except ImportError:
    # Hàm giả lập nếu không tìm thấy file utils (để tránh lỗi crash app)
    def load_model_resources(): return None, None
    def predict_debug(t, v, m): return 0.5, ["Error"], [0]
    def load_training_data_for_app(): return pd.DataFrame()

# ==========================================
# 👇 MAIN FUNCTION (Bắt buộc phải có hàm này)
# ==========================================
def show():
    # --- CSS STYLING (Chỉ áp dụng cho trang này) ---
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

    # Khởi tạo Session State cho input text
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ""

    # Chia cột giao diện
    col_main, col_sidebar = st.columns([2, 1])

    # --- 2. CỘT PHẢI: CÔNG CỤ DATA ---
    with col_sidebar:
        st.markdown("### 🎲 Dữ liệu mẫu")
        st.info("Lấy ngẫu nhiên 1 câu trong dữ liệu Training để test.")
        
        if st.button("🔄 Lấy mẫu ngẫu nhiên", use_container_width=True):
            df = load_training_data_for_app() 
            if not df.empty:
                sample = df.sample(1).iloc[0]
                st.session_state['input_text'] = sample['Content']
                st.session_state['true_label'] = sample['Label'] 
                st.toast(f"Đã lấy mẫu: {sample['Label']}", icon="✅")
            else:
                st.error("Không tìm thấy dữ liệu mẫu.")

        # Hiển thị nhãn gốc nếu có
        if 'true_label' in st.session_state and st.session_state['input_text']:
            st.caption(f"🏷️ Nhãn gốc: **{st.session_state['true_label']}**")

    # --- 3. CỘT TRÁI: PHÂN TÍCH ---
    with col_main:
        user_input = st.text_area("Nhập nội dung review:", 
                                  value=st.session_state['input_text'], 
                                  height=150,
                                  placeholder="Ví dụ: Sản phẩm dùng rất chán, phí tiền...")
        
        if st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True):
            if not model:
                st.error("⚠️ Chưa có Model! Vui lòng chạy file `train_pytorch.py` trước.")
            elif not user_input.strip():
                st.warning("Vui lòng nhập nội dung.")
            else:
                # --- GỌI HÀM DỰ ĐOÁN ---
                score, words, tokens = predict_debug(user_input, vocab, model)
                
                # --- HIỂN THỊ KẾT QUẢ ---
                st.divider()
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("#### Kết quả:")
                    if score >= 0.6:
                        st.markdown(f'<div class="result-box pos">TÍCH CỰC<br>{score:.2%}</div>', unsafe_allow_html=True)
                    elif score <= 0.4:
                        st.markdown(f'<div class="result-box neg">TIÊU CỰC<br>{(1-score):.2%}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box neu">TRUNG TÍNH<br>{score:.2f}</div>', unsafe_allow_html=True)
                
                with c2:
                    st.markdown("#### Độ tin cậy:")
                    st.progress(score)

                # --- CHI TIẾT TOKEN ---
                with st.expander("🔍 Chi tiết Tokenization (Máy đọc thế nào?)", expanded=True):
                    html_tokens = []
                    unk_count = 0
                    for w, idx in zip(words, tokens):
                        if idx == 0: # 0 là UNK (Unknown)
                            html_tokens.append(f'<span style="background-color:#ffcccc; color:red; padding:2px; border-radius:3px;">{w} (?)</span>')
                            unk_count += 1
                        else:
                            html_tokens.append(f'<span style="background-color:#e6ffe6; padding:2px; border-radius:3px;">{w}</span>')
                    
                    st.markdown(" ".join(html_tokens), unsafe_allow_html=True)
                    st.caption(f"UNK count: {unk_count} (Từ vựng máy chưa học).")

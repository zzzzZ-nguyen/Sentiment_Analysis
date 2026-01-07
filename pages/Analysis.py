import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import load_model_resources, predict, get_data_files, load_dataset

st.set_page_config(page_title="Analysis Debug", page_icon="🕵️", layout="wide")

st.title("🕵️ Phân tích & Gỡ lỗi (Debug)")

vocab, model = load_model_resources()
if not model: st.error("Chưa có model."); st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Nhập nội dung:", height=100)
    
    if st.button("🚀 Phân tích"):
        if user_input:
            # Gọi hàm predict mới (trả về cả thông tin debug)
            score, words_cleaned, token_ids = predict(user_input, vocab, model)
            
            # 1. KẾT QUẢ
            st.write("---")
            if score >= 0.6: st.success(f"TÍCH CỰC ({score:.2%})")
            elif score <= 0.4: st.error(f"TIÊU CỰC ({(1-score):.2%})")
            else: st.warning(f"TRUNG TÍNH ({score:.2f})")
            
            # 2. PHẦN DEBUG (QUAN TRỌNG)
            with st.expander("🔍 Tại sao ra kết quả này? (Xem chi tiết)", expanded=True):
                st.write("**1. Máy đã làm sạch văn bản như thế nào?**")
                st.code(str(words_cleaned))
                
                st.write("**2. Máy hiểu từ vựng ra sao? (0 là từ lạ)**")
                mapped_words = []
                unk_count = 0
                for w, idx in zip(words_cleaned, token_ids):
                    if idx == 0:
                        mapped_words.append(f"{w} (UNK ❌)")
                        unk_count += 1
                    else:
                        mapped_words.append(f"{w} ({idx} ✅)")
                
                st.write(f" -> Tỷ lệ từ lạ: {unk_count}/{len(words_cleaned)}")
                st.json(mapped_words)
                
                if unk_count > len(words_cleaned) / 2:
                    st.warning("⚠️ Cảnh báo: Quá nhiều từ lạ (UNK). Model gần như đang đoán mò. Hãy thêm dữ liệu train chứa các từ này.")

with col2:
    st.info("Mẹo: Nếu bạn thấy nhiều từ bị đánh dấu 'UNK ❌', nghĩa là lúc train chưa có từ đó. Bạn cần thêm dữ liệu vào file Excel/CSV và train lại.")

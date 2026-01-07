import streamlit as st
import torch
import torch.nn as nn
import pickle
import os
import re

# ==========================================
# 1. CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="Deep Learning Analysis", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    div.stButton > button {background-color: #2b6f3e; color: white; width: 100%;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ĐỊNH NGHĨA LẠI MODEL (Phải giống hệt file train)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden))
        return out

# ==========================================
# 3. HÀM LOAD MODEL & XỬ LÝ TEXT
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load Từ điển (Vocab)
    vocab_path = "models/vocab.pkl"
    if not os.path.exists(vocab_path):
        return None, None
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load Model (Trọng số)
    model_path = "models/sentiment_model.pth"
    if not os.path.exists(model_path):
        return None, vocab
        
    device = torch.device('cpu') # Streamlit Cloud dùng CPU
    
    # Khởi tạo lại kiến trúc model
    model = LSTMClassifier(len(vocab), 100, 128, 3)
    
    # Load trọng số đã train vào
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Chuyển sang chế độ dự đoán (không học nữa)
    return model, vocab

def text_to_tensor(text, vocab, max_len=20):
    # Xử lý text giống hệt lúc train
    words = text.lower().split()
    indices = [vocab.get(w, vocab.get('<UNK>', 1)) for w in words]
    
    # Padding
    if len(indices) < max_len:
        indices += [vocab.get('<PAD>', 0)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    return torch.tensor([indices], dtype=torch.long) # Thêm batch dimension [1, seq_len]

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.title("🧠 Sentiment Analysis (LSTM Model)")
st.write("Sử dụng mô hình Deep Learning (PyTorch) đã được huấn luyện trước.")

# Load model
try:
    model, vocab = load_artifacts()
    if model is None or vocab is None:
        st.error("⚠️ Không tìm thấy file model hoặc vocab trong thư mục `models/`. Vui lòng chạy `train_pytorch.py` trên máy local trước rồi upload file kết quả lên.")
        st.stop()
except Exception as e:
    st.error(f"Lỗi khi load model: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Nhập liệu")
    user_input = st.text_area("Nhập bình luận:", height=150, placeholder="Sản phẩm dùng rất tốt...")
    
    if st.button("🔍 Phân tích cảm xúc"):
        if user_input.strip():
            # Dự đoán
            tensor_input = text_to_tensor(user_input, vocab)
            with torch.no_grad():
                outputs = model(tensor_input)
                probs = torch.softmax(outputs, dim=1) # Chuyển thành xác suất
                max_prob, predicted_class = torch.max(probs, 1)
                
            prediction = predicted_class.item()
            confidence = max_prob.item()
            
            # Mapping kết quả
            labels = {0: "Negative (Tiêu cực)", 1: "Neutral (Trung tính)", 2: "Positive (Tích cực)"}
            result_text = labels[prediction]
            
            st.session_state['result'] = (result_text, confidence)
        else:
            st.warning("Vui lòng nhập nội dung!")

with col2:
    st.subheader("Kết quả")
    if 'result' in st.session_state:
        label, conf = st.session_state['result']
        
        if "Positive" in label:
            st.success(f"Dự đoán: **{label}**")
        elif "Negative" in label:
            st.error(f"Dự đoán: **{label}**")
        else:
            st.info(f"Dự đoán: **{label}**")
            
        st.metric("Độ tin cậy", f"{conf:.2%}")
        st.progress(conf)

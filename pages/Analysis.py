import streamlit as st
import torch
import torch.nn as nn
import pickle
import os
import re

# Định nghĩa lại class LSTM để load được model
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

@st.cache_resource
def load_pytorch_model():
    vocab_path = "models/vocab.pkl"
    model_path = "models/sentiment_model.pth"
    
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        return None, None
        
    # Load Vocab
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Init Model
    device = torch.device('cpu')
    model = LSTMClassifier(len(vocab), 100, 128, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, vocab

def text_to_tensor(text, vocab, max_len=50):
    words = text.lower().split()
    indices = [vocab.get(w, vocab.get('<UNK>', 1)) for w in words]
    if len(indices) < max_len:
        indices += [vocab.get('<PAD>', 0)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor([indices], dtype=torch.long)

# ==========================================
# MAIN FUNCTION (Được gọi từ app.py)
# ==========================================
def show():
    st.markdown("<h2 style='color:#2b6f3e;'>🧠 Phân Tích Cảm Xúc (Deep Learning)</h2>", unsafe_allow_html=True)
    st.write("Sử dụng mô hình LSTM đã huấn luyện để dự đoán cảm xúc (Tiếng Việt/Anh).")

    model, vocab = load_pytorch_model()
    
    if not model:
        st.warning("⚠️ Chưa tìm thấy model. Hãy chạy `python train_pytorch.py` trước!")
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Nhập bình luận sản phẩm:", height=150, placeholder="Ví dụ: Sản phẩm dùng rất tốt, giao hàng nhanh...")
        
        if st.button("🚀 Phân Tích Ngay", type="primary"):
            if user_input.strip():
                # Dự đoán
                tensor_input = text_to_tensor(user_input, vocab)
                with torch.no_grad():
                    outputs = model(tensor_input)
                    probs = torch.softmax(outputs, dim=1)
                    max_prob, predicted_class = torch.max(probs, 1)
                
                # Mapping kết quả (0: Neg, 1: Neu, 2: Pos - Dựa theo code train)
                labels = {0: "Tiêu cực (Negative)", 1: "Trung tính (Neutral)", 2: "Tích cực (Positive)"}
                colors = {0: "error", 1: "warning", 2: "success"}
                
                pred_label = labels[predicted_class.item()]
                conf = max_prob.item()
                
                # Hiển thị
                st.divider()
                msg_func = getattr(st, colors[predicted_class.item()])
                msg_func(f"Kết quả: **{pred_label}**")
                st.info(f"Độ tin cậy: **{conf:.2%}**")
            else:
                st.warning("Vui lòng nhập nội dung.")

    with col2:
        st.info("ℹ️ **Hướng dẫn:**\n\nNhập một câu bình luận về sản phẩm (điện thoại, máy tính, v.v.) để xem máy tính đánh giá cảm xúc như thế nào.")

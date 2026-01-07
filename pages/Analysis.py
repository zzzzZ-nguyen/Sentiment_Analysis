import streamlit as st
import os
import pickle
import numpy as np

# --- Cấu hình thư viện PyTorch ---
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==========================================
# 1. ĐỊNH NGHĨA LẠI MODEL (Bắt buộc phải giống lúc Train)
# ==========================================
if HAS_TORCH:
    class SentimentLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
            super(SentimentLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
            self.dropout = nn.Dropout(drop_prob)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, hidden):
            batch_size = x.size(0)
            embeds = self.embedding(x)
            lstm_out, hidden = self.lstm(embeds, hidden)
            lstm_out = lstm_out.contiguous().view(-1, hidden_dim)
            out = self.dropout(lstm_out)
            out = self.fc(out)
            out = self.sigmoid(out)
            out = out.view(batch_size, -1)
            out = out[:, -1]
            return out, hidden

        def init_hidden(self, batch_size, device):
            weight = next(self.parameters()).data
            hidden = (weight.new(n_layers, batch_size, hidden_dim).zero_().to(device),
                      weight.new(n_layers, batch_size, hidden_dim).zero_().to(device))
            return hidden
    
    # Cấu hình Hyperparameters (Phải khớp với file train)
    EMBEDDING_DIM = 400
    HIDDEN_DIM = 256 # Hoặc 128 tùy bạn chỉnh lúc train
    N_LAYERS = 2

# ==========================================
# 2. HÀM XỬ LÝ TEXT & LOAD MODEL
# ==========================================
def load_resources():
    # Load Vocab
    vocab_path = "models/vocab.pkl" # Đảm bảo bạn đã lưu file này lúc train
    model_path = "models/sentiment_model.pth" # Đảm bảo file này tồn tại
    
    vocab = None
    model = None
    
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            
    if HAS_TORCH and os.path.exists(model_path) and vocab:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = len(vocab) + 1
        model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS)
        # Load state dict
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except:
            model = None # Lỗi sai cấu trúc model
            
    return vocab, model

def predict_sentiment(text, vocab, model):
    if not vocab or not model:
        return None, 0.0

    # Tokenize
    words = text.split()
    review_int = []
    for word in words:
        review_int.append(vocab.get(word, 0)) # 0 là padding/unknown
    
    # Pad/Truncate về 50
    seq_len = 50
    if len(review_int) < seq_len:
        features = list(np.zeros(seq_len - len(review_int), dtype=int)) + review_int
    else:
        features = review_int[:seq_len]
    
    # Convert to Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tensor = torch.tensor([features], dtype=torch.long).to(device)
    h = model.init_hidden(1, device)
    
    # Predict
    with torch.no_grad():
        output, _ = model(feature_tensor, h)
        pred = output.item()
    
    return pred # Trả về giá trị 0.0 -> 1.0

# ==========================================
# 3. GIAO DIỆN CHÍNH (Hàm show)
# ==========================================
def show():
    # KHÔNG DÙNG st.set_page_config()
    
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("🧠 Phân Tích Cảm Xúc (Deep Learning)")
    st.write("Sử dụng mô hình LSTM đã huấn luyện để dự đoán bình luận mới.")

    if not HAS_TORCH:
        st.error("Chưa cài đặt PyTorch.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Load Model
    vocab, model = load_resources()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Nhập bình luận:")
        user_input = st.text_area("Nội dung đánh giá:", height=150, placeholder="Ví dụ: Sản phẩm dùng rất tốt, giao hàng nhanh...")
        
        btn_predict = st.button("🚀 Phân Tích Ngay", type="primary")
        
        if btn_predict:
            if not user_input.strip():
                st.warning("Vui lòng nhập nội dung!")
            elif model is None:
                st.error("⚠️ Chưa tìm thấy Model! Vui lòng vào trang 'Train PyTorch' để huấn luyện trước.")
            else:
                with st.spinner("Đang phân tích..."):
                    score = predict_sentiment(user_input, vocab, model)
                    
                    st.divider()
                    st.markdown("### Kết quả dự đoán:")
                    
                    # Logic hiển thị: < 0.4 là Negative, > 0.6 là Positive, ở giữa là Neutral
                    if score >= 0.6:
                        st.success(f"Dự đoán: **TÍCH CỰC (Positive)**")
                        st.metric("Độ tin cậy", f"{score:.2%}")
                    elif score <= 0.4:
                        st.error(f"Dự đoán: **TIÊU CỰC (Negative)**")
                        st.metric("Độ tin cậy", f"{(1-score):.2%}")
                    else:
                        st.warning(f"Dự đoán: **TRUNG TÍNH (Neutral)**")
                        st.metric("Điểm số", f"{score:.2f}")

    with col2:
        st.info("ℹ️ **Thông tin:**\n\nĐây là mô hình LSTM (Long Short-Term Memory) học trên mức độ từ (Word-level).\n\nKết quả trả về là xác suất (0-1):"
                "\n- Càng gần 1: Tích cực"
                "\n- Càng gần 0: Tiêu cực")
        
        if model:
            st.success("✅ Model đã được load thành công!")
        else:
            st.error("❌ Chưa có Model (Hãy Train trước)")

    st.markdown('</div>', unsafe_allow_html=True)

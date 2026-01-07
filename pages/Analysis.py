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
# 1. ĐỊNH NGHĨA KIẾN TRÚC MODEL
# (Bắt buộc phải KHỚP 100% với file Train)
# ==========================================
if HAS_TORCH:
    class SentimentLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
            super(SentimentLSTM, self).__init__()
            self.output_dim = output_dim
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
            self.dropout = nn.Dropout(drop_prob)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, hidden):
            batch_size = x.size(0)
            embeds = self.embedding(x)
            lstm_out, hidden = self.lstm(embeds, hidden)
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
            
            out = self.dropout(lstm_out)
            out = self.fc(out)
            out = self.sigmoid(out)
            
            out = out.view(batch_size, -1)
            out = out[:, -1]
            return out, hidden

        def init_hidden(self, batch_size, device):
            weight = next(self.parameters()).data
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
            return hidden
    
    # Cấu hình Hyperparameters (Phải khớp file Train)
    EMBEDDING_DIM = 400
    HIDDEN_DIM = 256 
    N_LAYERS = 2

# ==========================================
# 2. HÀM XỬ LÝ TEXT & LOAD MODEL
# ==========================================
@st.cache_resource
def load_pytorch_model():
    vocab_path = "models/vocab.pkl"
    model_path = "models/sentiment_model.pth"
    
    vocab = None
    model = None
    
    # 1. Load Vocab
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    
    # 2. Load Model Architecture & State
    if HAS_TORCH and os.path.exists(model_path) and vocab:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = len(vocab) + 1
        
        # Khởi tạo model
        model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS)
        
        try:
            # Load trọng số
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval() # Chế độ dự đoán
        except Exception as e:
            print(f"Lỗi load model: {e}")
            model = None 
            
    return vocab, model

def predict_sentiment(text, vocab, model):
    if not vocab or not model:
        return 0.5 

    # --- QUAN TRỌNG: Preprocessing ---
    # Phải lower() vì vocab lúc train là chữ thường
    words = text.lower().split()
    
    review_int = []
    for word in words:
        review_int.append(vocab.get(word, 0)) # 0 là từ lạ (unknown)
    
    # Padding / Truncating (Độ dài 50)
    seq_len = 50
    if len(review_int) < seq_len:
        features = list(np.zeros(seq_len - len(review_int), dtype=int)) + review_int
    else:
        features = review_int[:seq_len]
    
    # Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tensor = torch.tensor([features], dtype=torch.long).to(device)
    h = model.init_hidden(1, device)
    
    with torch.no_grad():
        output, _ = model(feature_tensor, h)
        pred = output.item()
    
    return pred

# ==========================================
# 3. GIAO DIỆN CHÍNH (Hàm Show)
# ==========================================
def show():
    # CSS
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #2b6f3e; color: white; border-radius: 5px; width: 100%;
        font-weight: bold;
    }
    .stTextArea textarea { background-color: #ffffff; color: #333; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#2b6f3e;'>🧠 Deep Learning Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.write("Sử dụng mô hình LSTM (PyTorch) đã được huấn luyện.")

    if not HAS_TORCH:
        st.error("⚠️ Thư viện `torch` chưa được cài đặt.")
        return

    # Load Model
    vocab, model = load_pytorch_model()

    if model is None:
        st.error("⚠️ Không tìm thấy Model! Hãy vào trang **Train PyTorch**, tạo dữ liệu mẫu và bấm Train trước.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Input Review")
        user_input = st.text_area("Nhập nội dung đánh giá (tiếng Việt):", height=150, placeholder="Ví dụ: Sản phẩm này dùng rất tốt, pin trâu...")
        
        if st.button("🚀 Analyze Sentiment"):
            if user_input.strip():
                # Dự đoán
                with st.spinner("Đang phân tích..."):
                    score = predict_sentiment(user_input, vocab, model)
                
                # Hiển thị
                st.write("---")
                st.markdown("### 🎯 Result")
                
                # Logic phân ngưỡng
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
        st.info("**Positive:**\n- Sản phẩm dùng rất tốt.\n- Giao hàng nhanh, đóng gói đẹp.")
        st.error("**Negative:**\n- Hàng kém chất lượng.\n- Mới dùng đã hỏng.")
        st.warning("**Neutral:**\n- Tạm được.\n- Cũng bình thường.")

    st.write("---")

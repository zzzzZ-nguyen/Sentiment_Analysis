import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
from collections import Counter

# --- Cấu hình trang (Chỉ chạy nếu file chạy độc lập) ---
# Nếu chạy qua app.py thì dòng này sẽ bị bỏ qua để tránh lỗi
try:
    if __name__ == "__main__":
        st.set_page_config(page_title="Train PyTorch", layout="wide")
except:
    pass

# --- Import PyTorch ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==========================================
# 1. ĐỊNH NGHĨA MODEL (Giống hệt bên Analysis)
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
            out = out[:, -1] # Lấy output của bước thời gian cuối cùng
            return out, hidden

        def init_hidden(self, batch_size, device):
            weight = next(self.parameters()).data
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
            return hidden

# ==========================================
# 2. HÀM XỬ LÝ DỮ LIỆU
# ==========================================
def read_txt(file_path):
    """Đọc file txt, mỗi dòng là một review"""
    reviews = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    reviews.append(line)
    return reviews

def preprocess_data():
    # 1. Đọc dữ liệu
    # Giả sử file nằm cùng cấp hoặc thư mục gốc. Bạn có thể sửa đường dẫn.
    pos_reviews = read_txt("train_positive_tokenized.txt")
    neg_reviews = read_txt("train_negative_tokenized.txt")
    
    # Lưu ý: Ta chỉ train Positive (1) và Negative (0) để model phân cực rõ ràng.
    # Neutral có thể bỏ qua hoặc gán 0.5 (nhưng gán 0/1 tốt hơn cho sigmoid).
    
    if not pos_reviews or not neg_reviews:
        return None, None, None, "Không tìm thấy file dữ liệu (train_positive/negative_tokenized.txt)"

    reviews = pos_reviews + neg_reviews
    labels = [1]*len(pos_reviews) + [0]*len(neg_reviews) # 1=Pos, 0=Neg

    # 2. Tạo Vocab
    words = []
    for r in reviews:
        words.extend(r.split())
    
    count_words = Counter(words)
    # Sắp xếp từ xuất hiện nhiều nhất
    sorted_words = count_words.most_common(len(count_words))
    
    # Mapping từ -> số (bắt đầu từ 1, 0 dành cho padding)
    vocab_to_int = {w: i+1 for i, (w, c) in enumerate(sorted_words)}
    
    # 3. Mã hóa reviews
    reviews_int = []
    for r in reviews:
        r_int = [vocab_to_int[w] for w in r.split()]
        reviews_int.append(r_int)
        
    # 4. Padding (Độ dài cố định 50)
    seq_len = 50
    features = np.zeros((len(reviews_int), seq_len), dtype=int)
    for i, row in enumerate(reviews_int):
        features[i, -len(row):] = np.array(row)[:seq_len]

    # Convert to Tensor
    X = torch.from_numpy(features)
    y = torch.from_numpy(np.array(labels)).float()

    return X, y, vocab_to_int, None

# ==========================================
# 3. GIAO DIỆN CHÍNH (Hàm show)
# ==========================================
def show():
    # CSS
    st.markdown("""
    <style>
    div.stButton > button {background-color: #ff4b4b; color: white; width: 100%;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("🔥 Huấn luyện Model LSTM (PyTorch)")
    st.write("Huấn luyện mô hình Deep Learning trên dữ liệu `positive` và `negative` của bạn.")

    if not HAS_TORCH:
        st.error("⚠️ Chưa cài đặt thư viện `torch`. Vui lòng chạy `pip install torch`.")
        return

    # Check file
    if not os.path.exists("train_positive_tokenized.txt"):
        st.warning("⚠️ Không tìm thấy file `train_positive_tokenized.txt`. Vui lòng upload file vào cùng thư mục.")
        st.stop()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Tham số")
        epochs = st.number_input("Epochs (Số vòng lặp)", 1, 50, 5)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        lr = st.select_slider("Learning Rate", options=[0.01, 0.005, 0.001, 0.0001], value=0.001)
        
        btn_train = st.button("🚀 Bắt đầu Train")

    with col2:
        st.subheader("📈 Tiến trình")
        log_area = st.empty()
        chart_loss = st.empty()
        status_text = st.empty()
        
        if btn_train:
            # 1. Chuẩn bị dữ liệu
            status_text.info("Đang xử lý dữ liệu...")
            X, y, vocab, err = preprocess_data()
            
            if err:
                st.error(err)
            else:
                # 2. Setup Training
                train_data = TensorDataset(X, y)
                train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                status_text.info(f"Đang chạy trên: **{device}** (Vocab size: {len(vocab)})")
                
                # Hyperparameters
                vocab_size = len(vocab) + 1
                embedding_dim = 400
                hidden_dim = 256
                n_layers = 2
                
                model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, 1, n_layers)
                model.to(device)
                
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # 3. Training Loop
                model.train()
                loss_history = []
                
                start_time = time.time()
                
                progress_bar = st.progress(0)
                
                for e in range(epochs):
                    h = model.init_hidden(batch_size, device)
                    avg_loss = []
                    
                    for inputs, labels in train_loader:
                        h = tuple([each.data for each in h])
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        model.zero_grad()
                        output, h = model(inputs, h)
                        loss = criterion(output, labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                        
                        avg_loss.append(loss.item())
                    
                    # Update UI sau mỗi Epoch
                    epoch_loss = np.mean(avg_loss)
                    loss_history.append(epoch_loss)
                    
                    chart_loss.line_chart(loss_history)
                    log_area.text(f"Epoch {e+1}/{epochs} | Loss: {epoch_loss:.5f}")
                    progress_bar.progress((e + 1) / epochs)
                
                # 4. Lưu Model
                if not os.path.exists("models"):
                    os.makedirs("models")
                
                torch.save(model.state_dict(), "models/sentiment_model.pth")
                with open("models/vocab.pkl", "wb") as f:
                    pickle.dump(vocab, f)
                
                total_time = time.time() - start_time
                st.balloons()
                status_text.success(f"✅ Huấn luyện xong trong {total_time:.2f} giây! Model đã lưu tại `models/sentiment_model.pth`")
                st.info("👉 Giờ bạn có thể chuyển sang trang **Analysis** để thử nghiệm.")

    st.markdown('</div>', unsafe_allow_html=True)

# Để chạy độc lập khi test
if __name__ == "__main__":
    show()

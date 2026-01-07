import streamlit as st
import os
import pickle
import numpy as np
import time
import sys
from collections import Counter

# Import Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    # Import Class SentimentLSTM và các tham số cấu hình từ model_utils
    from model_utils import SentimentLSTM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# Hàm tạo dữ liệu mẫu
def create_sample_data():
    pos_data = "sản phẩm tốt\ndùng rất thích\ngiao hàng nhanh\nchất lượng tuyệt vời\nđáng tiền\n" * 10
    neg_data = "sản phẩm tệ\ndùng mau hỏng\ngiao hàng chậm\nthái độ lồi lõm\nlãng phí tiền\n" * 10
    with open("train_positive_tokenized.txt", "w", encoding="utf-8") as f: f.write(pos_data)
    with open("train_negative_tokenized.txt", "w", encoding="utf-8") as f: f.write(neg_data)

# Hàm xử lý dữ liệu
def preprocess_data():
    def read_txt(path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return [line.strip() for line in f if line.strip()]
        return []

    pos = read_txt("train_positive_tokenized.txt")
    neg = read_txt("train_negative_tokenized.txt")
    
    if not pos or not neg: return None, None, None, "Thiếu dữ liệu."
    
    reviews = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)
    
    words = " ".join(reviews).split()
    count_words = Counter(words)
    vocab = {w: i+1 for i, (w, c) in enumerate(count_words.most_common())}
    
    reviews_int = [[vocab.get(w, 0) for w in r.split()] for r in reviews]
    
    seq_len = 50
    features = np.zeros((len(reviews_int), seq_len), dtype=int)
    for i, row in enumerate(reviews_int):
        features[i, -min(len(row), seq_len):] = np.array(row)[:seq_len]
        
    X = torch.from_numpy(features)
    y = torch.from_numpy(np.array(labels)).float()
    
    return X, y, vocab, None

def show():
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("🔥 Huấn luyện Model LSTM")

    if not HAS_DEPS:
        st.error("Thiếu thư viện hoặc file `model_utils.py`.")
        return

    # Kiểm tra dữ liệu
    if not (os.path.exists("train_positive_tokenized.txt") and os.path.exists("train_negative_tokenized.txt")):
        st.warning("⚠️ Chưa có dữ liệu.")
        if st.button("🛠️ Tạo Dữ liệu Mẫu"):
            create_sample_data()
            st.rerun()
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        epochs = st.number_input("Epochs", 1, 50, 10)
        btn_train = st.button("🚀 Train Model")

    with col2:
        log_area = st.empty()
        chart = st.empty()
        
        if btn_train:
            X, y, vocab, err = preprocess_data()
            if err: st.error(err); return
            
            # Setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_data = TensorDataset(X, y)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=4, drop_last=False)
            
            vocab_size = len(vocab) + 1
            model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            
            model.train()
            losses = []
            
            for e in range(epochs):
                h = model.init_hidden(4, device) # Init với batch size mặc định
                batch_losses = []
                for inputs, labels in train_loader:
                    curr_batch = inputs.size(0)
                    h = model.init_hidden(curr_batch, device) # Re-init đúng size thực tế
                    inputs, labels = inputs.to(device), labels.to(device)
                    model.zero_grad()
                    out, h = model(inputs, h)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                
                avg_loss = np.mean(batch_losses)
                losses.append(avg_loss)
                chart.line_chart(losses)
                log_area.text(f"Epoch {e+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Lưu model
            if not os.path.exists("models"): os.makedirs("models")
            torch.save(model.state_dict(), "models/sentiment_model.pth")
            with open("models/vocab.pkl", "wb") as f: pickle.dump(vocab, f)
            
            st.success("✅ Huấn luyện xong! Hãy qua trang Analysis để thử.")
            st.balloons()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()

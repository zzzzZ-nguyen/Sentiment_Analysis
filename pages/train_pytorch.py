import streamlit as st
import os
import pickle
import numpy as np
import sys
from collections import Counter

# Import Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from model_utils import SentimentLSTM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, generate_better_data
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

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
    
    # Tokenize
    words = " ".join(reviews).split()
    count_words = Counter(words)
    # Lọc bỏ từ xuất hiện quá ít (ít hơn 1 lần)
    sorted_words = count_words.most_common()
    vocab = {w: i+1 for i, (w, c) in enumerate(sorted_words)}
    
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
    st.title("🔥 Huấn luyện Model LSTM (Nâng cao)")
    st.info("Hệ thống sử dụng dữ liệu giả lập chất lượng cao để cải thiện độ chính xác.")

    if not HAS_DEPS:
        st.error("Lỗi: Không tìm thấy thư viện hoặc file model_utils.py")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. Dữ liệu")
        if st.button("♻️ Tạo mới dữ liệu (2000 câu)"):
            generate_better_data()
            st.success("Đã tạo 2000 câu mẫu đa dạng!")
            
        st.subheader("2. Huấn luyện")
        epochs = st.number_input("Epochs", 1, 50, 5) # Data nhiều thì giảm epoch xuống 5 là đủ demo
        batch_size = st.selectbox("Batch Size", [32, 64], index=0)
        btn_train = st.button("🚀 Bắt đầu Train")

    with col2:
        log_area = st.empty()
        chart = st.empty()
        
        if btn_train:
            # Check file
            if not os.path.exists("train_positive_tokenized.txt"):
                st.error("Chưa có dữ liệu. Hãy bấm 'Tạo mới dữ liệu' trước.")
                return

            X, y, vocab, err = preprocess_data()
            if err: st.error(err); return
            
            # Setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_data = TensorDataset(X, y)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
            
            vocab_size = len(vocab) + 1
            model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            losses = []
            
            progress = st.progress(0)
            
            for e in range(epochs):
                h = model.init_hidden(batch_size, device)
                batch_losses = []
                for inputs, labels in train_loader:
                    h = tuple([each.data for each in h])
                    inputs, labels = inputs.to(device), labels.to(device)
                    model.zero_grad()
                    out, h = model(inputs, h)
                    loss = criterion(out, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    batch_losses.append(loss.item())
                
                avg_loss = np.mean(batch_losses)
                losses.append(avg_loss)
                chart.line_chart(losses)
                log_area.text(f"Epoch {e+1}/{epochs} | Loss: {avg_loss:.4f}")
                progress.progress((e+1)/epochs)
            
            # Lưu model
            if not os.path.exists("models"): os.makedirs("models")
            torch.save(model.state_dict(), "models/sentiment_model.pth")
            with open("models/vocab.pkl", "wb") as f: pickle.dump(vocab, f)
            
            st.success("✅ Huấn luyện xong! Model đã học được nhiều từ vựng hơn.")
            st.balloons()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()

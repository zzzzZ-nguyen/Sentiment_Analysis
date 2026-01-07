import streamlit as st
import pandas as pd
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Kiểm tra thư viện PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==========================================
# 1. CẤU HÌNH MODEL LSTM (PyTorch)
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
            out = out[:, -1] # Lấy output của time-step cuối cùng
            return out, hidden

        def init_hidden(self, batch_size, device):
            weight = next(self.parameters()).data
            hidden = (weight.new(n_layers, batch_size, hidden_dim).zero_().to(device),
                      weight.new(n_layers, batch_size, hidden_dim).zero_().to(device))
            return hidden

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (HELPER FUNCTIONS)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    data_dir = "data"
    data = []
    
    # Mapping nhãn về số: Negative=0, Neutral=1, Positive=2 (Tuy nhiên LSTM binary thường chỉ train 0/1, ở đây ta demo đơn giản gộp Neutral)
    # Để đơn giản cho demo Binary Classification: Gom Neutral & Positive = 1, Negative = 0
    
    files = {
        0: "train_negative_tokenized.txt",
        1: "train_positive_tokenized.txt" 
        # Tạm bỏ qua Neutral để train Binary cho model đơn giản, hoặc gộp Neutral vào Positive
    }

    if not os.path.exists(data_dir):
        return None, None

    for label, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        data.append((line.strip(), label))
    
    if not data:
        return None, None

    df = pd.DataFrame(data, columns=['text', 'label'])
    return df

def build_vocab(sentences):
    # Tạo bộ từ điển đơn giản
    word_list = " ".join(sentences).split()
    from collections import Counter
    counts = Counter(word_list)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab_to_int

def tokenize_review(review, vocab_to_int, seq_len=50):
    # Chuyển text thành list số
    review_int = []
    for word in review.split():
        if word in vocab_to_int:
            review_int.append(vocab_to_int[word])
        else:
            review_int.append(0) # 0 cho từ không có trong vocab
    
    # Padding / Truncating
    if len(review_int) < seq_len:
        features = list(np.zeros(seq_len - len(review_int), dtype=int)) + review_int
    else:
        features = review_int[:seq_len]
    
    return np.array(features)

# ==========================================
# 3. GIAO DIỆN CHÍNH (HÀM SHOW)
# ==========================================
def show():
    # --- CSS Styles ---
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #E58E61; color: white; border-radius: 8px; width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("🔥 Huấn luyện mô hình Deep Learning (LSTM)")
    
    if not HAS_TORCH:
        st.error("⚠️ Thư viện `torch` chưa được cài đặt. Vui lòng chạy `pip install torch`.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Cấu hình Hyperparameters")
        n_epochs = st.number_input("Số Epochs (Vòng lặp)", min_value=1, max_value=50, value=5)
        lr = st.select_slider("Learning Rate", options=[0.01, 0.005, 0.001, 0.0001], value=0.001)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        hidden_dim = st.selectbox("Hidden Dimension", [128, 256, 512], index=1)
        embedding_dim = 400
        
        st.info("Mô hình: LSTM (Long Short-Term Memory)\n\nInput: Word Embeddings\n\nOutput: Binary (Pos/Neg)")
        
        start_btn = st.button("🚀 Bắt đầu Huấn luyện")

    with col2:
        st.subheader("📈 Kết quả Training")
        status_text = st.empty()
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        
        # Load data
        df = load_and_preprocess_data()
        
        if df is None:
            st.warning("Chưa tìm thấy dữ liệu trong folder `data/`.")
        else:
            st.write(f"Dữ liệu sẵn sàng: {len(df)} mẫu (Negative & Positive)")

            if start_btn:
                # 1. Chuẩn bị dữ liệu
                status_text.text("⏳ Đang xử lý dữ liệu & Xây dựng từ điển...")
                vocab_to_int = build_vocab(df['text'].tolist())
                vocab_size = len(vocab_to_int) + 1
                
                # Tokenize toàn bộ dataset
                seq_len = 50
                features = np.array([tokenize_review(r, vocab_to_int, seq_len) for r in df['text']])
                labels = np.array(df['label'].tolist())
                
                # Split Train/Val (80/20)
                split_idx = int(len(features) * 0.8)
                train_x, val_x = features[:split_idx], features[split_idx:]
                train_y, val_y = labels[:split_idx], labels[split_idx:]
                
                # Tạo TensorDataset
                train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
                train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
                
                # 2. Khởi tạo Model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                status_text.text(f"🖥️ Đang Training trên thiết bị: {device}")
                
                model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, 1, 2).to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # 3. Training Loop
                losses = []
                model.train()
                
                for epoch in range(n_epochs):
                    h = model.init_hidden(batch_size, device)
                    avg_loss = 0
                    counter = 0
                    
                    for inputs, labels in train_loader:
                        counter += 1
                        h = tuple([each.data for each in h])
                        inputs, labels = inputs.to(device), labels.float().to(device)
                        
                        model.zero_grad()
                        output, h = model(inputs, h)
                        loss = criterion(output, labels)
                        loss.backward()
                        optimizer.step()
                        
                        avg_loss += loss.item()
                    
                    # Cập nhật kết quả sau mỗi epoch
                    curr_loss = avg_loss / len(train_loader)
                    losses.append(curr_loss)
                    
                    # Update UI
                    progress = (epoch + 1) / n_epochs
                    progress_bar.progress(progress)
                    status_text.markdown(f"**Epoch {epoch+1}/{n_epochs}** - Loss: `{curr_loss:.4f}`")
                    
                    # Vẽ biểu đồ Loss realtime
                    chart_data = pd.DataFrame(losses, columns=["Training Loss"])
                    chart_placeholder.line_chart(chart_data)
                    
                    time.sleep(0.1) # Delay nhẹ để UI mượt hơn

                st.success("🎉 Huấn luyện hoàn tất!")
                st.balloons()

    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
from collections import Counter

# --- Cấu hình trang (Chỉ chạy nếu file chạy độc lập) ---
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
# 1. ĐỊNH NGHĨA MODEL (SentimentLSTM)
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

# ==========================================
# 2. HÀM TẠO DỮ LIỆU MẪU (FIX LỖI THIẾU FILE)
# ==========================================
def create_sample_data():
    """Tạo file dữ liệu mẫu nếu chưa có"""
    # Dữ liệu Tích cực mẫu
    pos_data = """sản phẩm dùng rất tốt
chất lượng tuyệt vời giao hàng nhanh
tôi rất thích sản phẩm này
đóng gói cẩn thận đẹp mắt
dùng rất bền đáng đồng tiền
nhân viên tư vấn nhiệt tình
mọi người nên mua nhé
hàng chính hãng chất lượng cao
sử dụng mượt mà không lỗi lầm
đánh giá 5 sao cho shop
"""
    # Dữ liệu Tiêu cực mẫu
    neg_data = """sản phẩm quá tệ
dùng được vài hôm đã hỏng
giao hàng chậm chạp thái độ lồi lõm
hàng giả không giống hình
đừng mua phí tiền
chất lượng kém quá thất vọng
gọi hỗ trợ không ai nghe máy
đóng gói sơ sài bị vỡ
quảng cáo sai sự thật
trải nghiệm tồi tệ
"""
    
    with open("train_positive_tokenized.txt", "w", encoding="utf-8") as f:
        f.write(pos_data)
        
    with open("train_negative_tokenized.txt", "w", encoding="utf-8") as f:
        f.write(neg_data)

# ==========================================
# 3. HÀM XỬ LÝ DỮ LIỆU
# ==========================================
def read_txt(file_path):
    reviews = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    reviews.append(line)
    return reviews

def preprocess_data():
    # Đọc dữ liệu
    pos_reviews = read_txt("train_positive_tokenized.txt")
    neg_reviews = read_txt("train_negative_tokenized.txt")
    
    if not pos_reviews or not neg_reviews:
        return None, None, None, "Dữ liệu rỗng."

    reviews = pos_reviews + neg_reviews
    labels = [1]*len(pos_reviews) + [0]*len(neg_reviews)

    # Tạo Vocab
    words = []
    for r in reviews:
        words.extend(r.split())
    
    count_words = Counter(words)
    sorted_words = count_words.most_common(len(count_words))
    vocab_to_int = {w: i+1 for i, (w, c) in enumerate(sorted_words)}
    
    # Mã hóa reviews
    reviews_int = []
    for r in reviews:
        r_int = [vocab_to_int.get(w, 0) for w in r.split()] # Dùng .get để tránh lỗi key
        reviews_int.append(r_int)
        
    # Padding
    seq_len = 50
    features = np.zeros((len(reviews_int), seq_len), dtype=int)
    for i, row in enumerate(reviews_int):
        if len(row) > 0:
            features[i, -min(len(row), seq_len):] = np.array(row)[:seq_len]

    # Tensor
    X = torch.from_numpy(features)
    y = torch.from_numpy(np.array(labels)).float()

    return X, y, vocab_to_int, None

# ==========================================
# 4. GIAO DIỆN CHÍNH (Hàm show)
# ==========================================
def show():
    st.markdown("""
    <style>
    div.stButton > button {background-color: #ff4b4b; color: white; width: 100%; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("🔥 Huấn luyện Model LSTM (PyTorch)")

    if not HAS_TORCH:
        st.error("⚠️ Chưa cài đặt thư viện `torch`. Vui lòng chạy `pip install torch`.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # --- KIỂM TRA & TẠO DATA ---
    file_exists = os.path.exists("train_positive_tokenized.txt") and os.path.exists("train_negative_tokenized.txt")
    
    if not file_exists:
        st.warning("⚠️ Chưa tìm thấy dữ liệu huấn luyện.")
        st.info("Bạn có muốn tạo dữ liệu mẫu (Sample Data) để chạy thử không?")
        
        if st.button("🛠️ Tạo Dữ Liệu Mẫu & Tiếp Tục"):
            create_sample_data()
            st.success("✅ Đã tạo file thành công! Vui lòng đợi trang tải lại...")
            time.sleep(1)
            st.rerun() # Tự động load lại trang
        
        st.markdown('</div>', unsafe_allow_html=True)
        return # Dừng hàm tại đây nếu chưa có file

    # --- NẾU ĐÃ CÓ DATA THÌ HIỆN GIAO DIỆN TRAIN ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Tham số")
        epochs = st.number_input("Epochs", 1, 100, 10) # Tăng default epoch lên 10 vì data ít
        batch_size = st.selectbox("Batch Size", [2, 4, 16, 32], index=1) # Giảm batch size vì data mẫu ít
        lr = st.select_slider("Learning Rate", options=[0.01, 0.005, 0.001], value=0.005)
        
        btn_train = st.button("🚀 Bắt đầu Train")

    with col2:
        st.subheader("📈 Tiến trình")
        log_area = st.empty()
        chart_loss = st.empty()
        status_text = st.empty()
        
        if btn_train:
            status_text.info("🔄 Đang xử lý dữ liệu...")
            X, y, vocab, err = preprocess_data()
            
            if err:
                st.error(err)
            else:
                # Setup Training
                # Data mẫu ít nên batch_size phải nhỏ hơn len(data)
                curr_batch = min(batch_size, len(X))
                
                train_data = TensorDataset(X, y)
                train_loader = DataLoader(train_data, shuffle=True, batch_size=curr_batch, drop_last=False)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                status_text.info(f"💻 Device: **{device}** | Vocab: {len(vocab)} từ | Samples: {len(X)}")
                
                # Model Init
                vocab_size = len(vocab) + 1
                embedding_dim = 400
                hidden_dim = 256
                n_layers = 2
                
                model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, 1, n_layers)
                model.to(device)
                
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Loop
                model.train()
                loss_history = []
                progress_bar = st.progress(0)
                
                start_time = time.time()
                
                for e in range(epochs):
                    h = model.init_hidden(curr_batch, device)
                    avg_loss = []
                    
                    for inputs, labels in train_loader:
                        # Handle batch size dynamic (nếu batch cuối lẻ)
                        current_batch_size = inputs.size(0)
                        h = model.init_hidden(current_batch_size, device) # Re-init hidden với đúng kích thước batch
                        
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        model.zero_grad()
                        output, h = model(inputs, h)
                        
                        loss = criterion(output, labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                        
                        avg_loss.append(loss.item())
                    
                    epoch_loss = np.mean(avg_loss) if avg_loss else 0
                    loss_history.append(epoch_loss)
                    
                    chart_loss.line_chart(loss_history)
                    log_area.text(f"Epoch {e+1}/{epochs} | Loss: {epoch_loss:.5f}")
                    progress_bar.progress((e + 1) / epochs)
                
                # Save
                if not os.path.exists("models"):
                    os.makedirs("models")
                
                torch.save(model.state_dict(), "models/sentiment_model.pth")
                with open("models/vocab.pkl", "wb") as f:
                    pickle.dump(vocab, f)
                
                st.balloons()
                status_text.success(f"✅ Xong! Model đã lưu vào `models/`.")
                st.info("👉 Hãy qua trang **Analysis** để test thử.")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import pickle
from collections import Counter

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
DATA_PATH = "data.csv"      # Tên file dữ liệu của bạn (nếu có)
MODEL_SAVE_PATH = "models/sentiment_model.pth" # Nơi lưu model
VOCAB_SAVE_PATH = "models/vocab.pkl"           # Nơi lưu bộ từ điển
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3              # 3 lớp: Negative, Neutral, Positive
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 16

# Tạo thư mục models nếu chưa có
if not os.path.exists("models"):
    os.makedirs("models")

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU (DATA PROCESSING)
# ==========================================
def load_data():
    # Nếu có file data.csv thì đọc, không thì tạo dữ liệu mẫu để test
    if os.path.exists(DATA_PATH):
        print(f"--- Đang đọc dữ liệu từ {DATA_PATH} ---")
        df = pd.read_csv(DATA_PATH)
        # Giả sử file csv có cột 'text' và 'label' (0, 1, 2)
        texts = df['text'].values
        labels = df['label'].values
    else:
        print("--- Không tìm thấy data.csv, đang tạo dữ liệu mẫu... ---")
        texts = [
            "sản phẩm rất tệ", "không thích cái này", "quá thất vọng", "hàng dởm", # Negative (0)
            "bình thường", "tạm được", "giống mô tả", "không có gì đặc biệt",      # Neutral (1)
            "tuyệt vời", "rất thích", "chất lượng cao", "đáng tiền mua"            # Positive (2)
        ]
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    return texts, labels

# Hàm xây dựng bộ từ điển (Tokenization đơn giản)
def build_vocab(texts):
    words = []
    for text in texts:
        words.extend(text.lower().split())
    count = Counter(words)
    # Sắp xếp theo tần suất xuất hiện
    vocab = {word: i+1 for i, (word, _) in enumerate(count.most_common())}
    vocab['<PAD>'] = 0 # Padding (đệm cho câu ngắn)
    vocab['<UNK>'] = len(vocab) # Unknown (từ lạ)
    return vocab

# Hàm chuyển câu thành số (Vectorization)
def text_to_indices(text, vocab, max_len=20):
    words = text.lower().split()
    indices = [vocab.get(w, vocab['<UNK>']) for w in words]
    # Padding hoặc cắt ngắn (Truncate)
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_indices = text_to_indices(self.texts[idx], self.vocab)
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 3. XÂY DỰNG MODEL (LSTM)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional=True giúp model đọc hiểu 2 chiều (trái qua phải, phải qua trái)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Nhân 2 vì là Bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Lấy trạng thái ẩn cuối cùng của 2 chiều ghép lại
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden))
        return out

# ==========================================
# 4. QUÁ TRÌNH HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
def train():
    # 1. Load data
    texts, labels = load_data()
    
    # 2. Build Vocab
    vocab = build_vocab(texts)
    print(f"Kích thước bộ từ điển: {len(vocab)}")
    
    # 3. Tạo DataLoader
    dataset = SentimentDataset(texts, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Khởi tạo Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")
    
    model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()       # Xóa gradient cũ
            outputs = model(inputs)     # Lan truyền xuôi (Forward)
            loss = criterion(outputs, targets) # Tính lỗi
            loss.backward()             # Lan truyền ngược (Backward)
            optimizer.step()            # Cập nhật trọng số
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(dataloader):.4f} - Acc: {100 * correct / total:.2f}%")

    # ==========================================
    # 5. LƯU MODEL VÀ TỪ ĐIỂN
    # ==========================================
    print("--- Đang lưu model... ---")
    
    # Lưu trọng số model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Lưu vocab (Rất quan trọng để dùng lại khi dự đoán)
    with open(VOCAB_SAVE_PATH, 'wb') as f:
        pickle.dump(vocab, f)
        
    print(f"Đã lưu model tại: {MODEL_SAVE_PATH}")
    print(f"Đã lưu vocab tại: {VOCAB_SAVE_PATH}")

if __name__ == "__main__":
    train()

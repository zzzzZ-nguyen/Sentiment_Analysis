import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from collections import Counter

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
# Tên file dữ liệu bạn đã upload (để cùng thư mục với file code này)
FILES = {
    0: "train_negative_tokenized.txt", # Nhãn 0: Negative
    1: "train_neutral_tokenized.txt",  # Nhãn 1: Neutral
    2: "train_positive_tokenized.txt"  # Nhãn 2: Positive
}

MODEL_SAVE_PATH = "models/sentiment_model.pth"
VOCAB_SAVE_PATH = "models/vocab.pkl"

EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3
LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 32

# Tạo thư mục models
if not os.path.exists("models"):
    os.makedirs("models")

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (ĐỌC FILE TXT)
# ==========================================
def read_data():
    texts = []
    labels = []
    
    print("--- Đang đọc dữ liệu từ các file TXT... ---")
    for label, filepath in FILES.items():
        if os.path.exists(filepath):
            count = 0
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        labels.append(label)
                        count += 1
            print(f"✅ Đã đọc {filepath}: {count} dòng.")
        else:
            print(f"⚠️ Không tìm thấy file: {filepath} (Bỏ qua)")
    
    return texts, labels

def build_vocab(texts):
    print("--- Đang xây dựng bộ từ điển... ---")
    words = []
    for text in texts:
        words.extend(text.lower().split())
    
    # Chỉ lấy từ xuất hiện > 1 lần để giảm nhiễu
    count = Counter(words)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, c in count.most_common():
        if c > 1: 
            vocab[word] = idx
            idx += 1
    return vocab

def text_to_indices(text, vocab, max_len=50):
    words = text.lower().split()
    indices = [vocab.get(w, vocab['<UNK>']) for w in words]
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
        vec = text_to_indices(self.texts[idx], self.vocab)
        return torch.tensor(vec, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 3. MÔ HÌNH LSTM (PYTORCH)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden))
        return out

# ==========================================
# 4. CHẠY HUẤN LUYỆN
# ==========================================
def train():
    # 1. Load Data
    texts, labels = read_data()
    if not texts:
        print("❌ Lỗi: Không có dữ liệu. Hãy đảm bảo các file .txt nằm cùng thư mục với file code này.")
        return

    # 2. Build Vocab
    vocab = build_vocab(texts)
    print(f"📖 Kích thước từ điển: {len(vocab)} từ")

    # 3. Prepare Model
    dataset = SentimentDataset(texts, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Bắt đầu train trên: {device}")
    
    model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(dataloader):.4f} | Acc: {100*correct/total:.2f}%")

    # 5. Save Results
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(VOCAB_SAVE_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✅ Đã lưu model mới tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

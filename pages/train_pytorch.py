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
DATA_DIR = "data"  # Tên thư mục chứa dữ liệu

# Đường dẫn các file Train
TRAIN_FILES = {
    0: os.path.join(DATA_DIR, "train_negative_tokenized.txt"),
    1: os.path.join(DATA_DIR, "train_neutral_tokenized.txt"),
    2: os.path.join(DATA_DIR, "train_positive_tokenized.txt")
}

# Đường dẫn file Test (Để kiểm tra độ chính xác)
TEST_FILE = os.path.join(DATA_DIR, "test_tokenized_ANS.txt")

# Nơi lưu model
MODEL_SAVE_PATH = "models/sentiment_model.pth"
VOCAB_SAVE_PATH = "models/vocab.pkl"

# Hyperparameters
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3
LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 32

# Tạo thư mục models nếu chưa có
if not os.path.exists("models"):
    os.makedirs("models")

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
def read_train_data():
    """Đọc 3 file train riêng biệt"""
    texts = []
    labels = []
    print("\n--- 1. Đang đọc dữ liệu huấn luyện (Train) ---")
    for label, filepath in TRAIN_FILES.items():
        if os.path.exists(filepath):
            count = 0
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        labels.append(label)
                        count += 1
            print(f"   - Đã đọc {os.path.basename(filepath)}: {count} dòng.")
        else:
            print(f"   ⚠️ CẢNH BÁO: Không tìm thấy file {filepath}")
    return texts, labels

def read_test_data():
    """Đọc file test đặc biệt (Dòng 1: Text, Dòng 2: Label)"""
    texts = []
    labels = []
    print("\n--- 2. Đang đọc dữ liệu kiểm thử (Test) ---")
    
    if os.path.exists(TEST_FILE):
        with open(TEST_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # File test có dạng: Dòng chẵn là Text, Dòng lẻ là Label (NEG, POS, NEU)
        for i in range(0, len(lines) - 1, 2):
            text = lines[i].strip()
            label_str = lines[i+1].strip()
            
            # Chuyển label chữ sang số
            if label_str == 'NEG': label = 0
            elif label_str == 'NEU': label = 1
            elif label_str == 'POS': label = 2
            else: continue # Bỏ qua nếu lỗi
            
            if text:
                texts.append(text)
                labels.append(label)
        print(f"   - Đã đọc file Test: {len(texts)} dòng.")
    else:
        print(f"   ⚠️ Không tìm thấy file Test tại {TEST_FILE}")
        
    return texts, labels

def build_vocab(texts):
    print("\n--- 3. Đang xây dựng bộ từ điển ---")
    words = []
    for text in texts:
        words.extend(text.lower().split())
    
    count = Counter(words)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, c in count.most_common():
        if c > 1: # Chỉ lấy từ xuất hiện > 1 lần
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
# 3. MÔ HÌNH LSTM
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
        # Ghép hidden state của 2 chiều
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden))
        return out

# ==========================================
# 4. CHẠY HUẤN LUYỆN & ĐÁNH GIÁ
# ==========================================
def train():
    # Load Data
    train_texts, train_labels = read_train_data()
    test_texts, test_labels = read_test_data()
    
    if not train_texts:
        print("❌ Lỗi: Không có dữ liệu Train.")
        return

    # Build Vocab (Dựa trên cả tập Train và Test để không bị sót từ)
    vocab = build_vocab(train_texts + test_texts)
    print(f"   - Kích thước từ điển: {len(vocab)} từ")

    # Prepare Datasets
    train_dataset = SentimentDataset(train_texts, train_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Bắt đầu Train trên thiết bị: {device}")
    
    model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- TRAINING LOOP ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
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
            
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")

    # --- EVALUATION ON TEST SET (Quan trọng) ---
    print("\n--- 📊 Đánh giá trên tập Test (Dữ liệu chưa từng học) ---")
    if test_texts:
        test_dataset = SentimentDataset(test_texts, test_labels, vocab)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model.eval() # Chuyển sang chế độ chấm điểm
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
        
        print(f"🎯 ĐỘ CHÍNH XÁC THỰC TẾ (TEST ACCURACY): {100 * test_correct / test_total:.2f}%")
    else:
        print("⚠️ Không có file test để đánh giá.")

    # Save Model
    print("\n--- 💾 Đang lưu model... ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(VOCAB_SAVE_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✅ Đã xong! File model tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

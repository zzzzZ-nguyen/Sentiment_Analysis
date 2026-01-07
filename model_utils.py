import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import random

# Cấu hình Hyperparameters
EMBEDDING_DIM = 400
HIDDEN_DIM = 256
N_LAYERS = 2

# 1. ĐỊNH NGHĨA MODEL
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

# 2. HÀM TẠO DỮ LIỆU ĐA DẠNG (NEW)
def generate_better_data():
    """Tạo 2000 dòng dữ liệu tích cực và tiêu cực từ kho từ vựng"""
    
    # Kho từ vựng Tích cực
    pos_adj = ["tốt", "tuyệt vời", "xuất sắc", "đẹp", "xịn", "bền", "nhanh", "thích", "ưng ý", "hài lòng", "đỉnh", "chất lượng"]
    pos_nouns = ["sản phẩm", "hàng", "shop", "dịch vụ", "đóng gói", "máy", "thiết bị", "chất liệu", "giao hàng"]
    pos_adv = ["rất", "quá", "cực kỳ", "vô cùng", "siêu", "thật sự"]
    
    # Kho từ vựng Tiêu cực
    neg_adj = ["tệ", "kém", "xấu", "hỏng", "chậm", "đắt", "lởm", "như hạch", "thất vọng", "phí tiền", "vỡ", "nát"]
    neg_nouns = ["sản phẩm", "hàng", "shop", "thái độ", "đóng gói", "vận chuyển", "nhân viên", "màn hình", "pin"]
    neg_adv = ["quá", "rất", "cực kỳ", "thật", "hơi"]

    pos_reviews = []
    neg_reviews = []

    # Tạo 1000 câu tích cực
    for _ in range(1000):
        structure = random.choice([1, 2, 3])
        if structure == 1: # "sản phẩm" + "rất" + "tốt"
            sentence = f"{random.choice(pos_nouns)} {random.choice(pos_adv)} {random.choice(pos_adj)}"
        elif structure == 2: # "rất" + "thích" + "sản phẩm"
            sentence = f"{random.choice(pos_adv)} {random.choice(['thích', 'yêu', 'ưng'])} {random.choice(pos_nouns)}"
        else: # "giao hàng" + "nhanh"
            sentence = f"{random.choice(pos_nouns)} {random.choice(pos_adj)}"
        pos_reviews.append(sentence)

    # Tạo 1000 câu tiêu cực
    for _ in range(1000):
        structure = random.choice([1, 2, 3])
        if structure == 1: # "hàng" + "quá" + "tệ"
            sentence = f"{random.choice(neg_nouns)} {random.choice(neg_adv)} {random.choice(neg_adj)}"
        elif structure == 2: # "đừng" + "mua"
            sentence = f"đừng {random.choice(['mua', 'tin', 'dùng'])} {random.choice(neg_nouns)}"
        else: # "chất lượng" + "kém"
            sentence = f"chất lượng {random.choice(neg_adj)}"
        neg_reviews.append(sentence)

    # Lưu file
    with open("train_positive_tokenized.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pos_reviews))
    with open("train_negative_tokenized.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(neg_reviews))
    
    return True

# 3. CÁC HÀM TIỆN ÍCH KHÁC
def load_model_resources():
    vocab_path = "models/vocab.pkl"
    model_path = "models/sentiment_model.pth"
    
    vocab = None
    model = None
    
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            
    if os.path.exists(model_path) and vocab:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = len(vocab) + 1
        model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error: {e}")
            model = None
    return vocab, model

def predict(text, vocab, model):
    if not vocab or not model: return 0.5
    
    # Xử lý text kỹ hơn
    words = text.lower().replace('.', '').replace(',', '').split()
    review_int = [vocab.get(w, 0) for w in words]
    
    # Nếu toàn từ lạ (toàn số 0) -> Trả về trung tính
    if sum(review_int) == 0:
        return 0.5
    
    seq_len = 50
    if len(review_int) < seq_len:
        features = list(np.zeros(seq_len - len(review_int), dtype=int)) + review_int
    else:
        features = review_int[:seq_len]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tensor = torch.tensor([features], dtype=torch.long).to(device)
    h = model.init_hidden(1, device)
    
    with torch.no_grad():
        output, _ = model(feature_tensor, h)
        pred = output.item()
        
    return pred

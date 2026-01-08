import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import pandas as pd
import re

# --- 1. CẤU HÌNH MODEL ---
EMBEDDING_DIM = 400
HIDDEN_DIM = 256
N_LAYERS = 2

# --- 2. HÀM CHUẨN HÓA TEXT (QUAN TRỌNG) ---
def clean_text(text):
    """Làm sạch văn bản đồng bộ cho cả Train và Test"""
    if not isinstance(text, str): return []
    text = text.lower()
    # Giữ lại chữ cái, số và dấu cách
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

# --- 3. CLASS MODEL LSTM ---
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

# --- 4. HÀM LOAD MODEL & DỮ LIỆU (MỚI) ---
def load_model_resources():
    vocab_path = "models/vocab.pkl"
    model_path = "models/sentiment_model.pth"
    vocab, model = None, None
    
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f: vocab = pickle.load(f)
            
    if os.path.exists(model_path) and vocab:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = len(vocab) + 1
        model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except: model = None
    return vocab, model

def load_training_data_for_app():
    """Đọc dữ liệu Training (TXT hoặc CSV) để dùng chung"""
    data_dir = "data"
    all_data = []
    
    # Ưu tiên 1: Đọc các file TXT chuẩn (như trong Training_Info)
    train_files = {
        "Negative": "train_negative_tokenized.txt",
        "Neutral": "train_neutral_tokenized.txt",
        "Positive": "train_positive_tokenized.txt"
    }
    
    found_txt = False
    if os.path.exists(data_dir):
        for label, filename in train_files.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                found_txt = True
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            all_data.append({"Content": line.strip(), "Label": label})
    
    # Ưu tiên 2: Nếu không có TXT, tìm file CSV/Excel bất kỳ
    if not found_txt and os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx'))]
        for f in files:
            try:
                path = os.path.join(data_dir, f)
                df = pd.read_csv(path) if f.endswith('.csv') else pd.read_excel(path)
                # Tìm cột text
                text_cols = [c for c in df.columns if df[c].dtype == object]
                if text_cols:
                    col_name = text_cols[0] # Lấy cột đầu tiên
                    for val in df[col_name].dropna():
                        all_data.append({"Content": str(val), "Label": "Unknown"})
            except: pass

    return pd.DataFrame(all_data)

# --- 5. HÀM DỰ ĐOÁN (TRẢ VỀ CHI TIẾT) ---
def predict_debug(text, vocab, model):
    if not vocab or not model: return 0.5, [], []
    
    words = clean_text(text)
    review_int = [vocab.get(w, 0) for w in words]
    
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
        
    return pred, words, review_int

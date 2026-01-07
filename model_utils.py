import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import pandas as pd
import re

# --- CẤU HÌNH ---
EMBEDDING_DIM = 400
HIDDEN_DIM = 256
N_LAYERS = 2

# 1. HÀM CHUẨN HÓA VĂN BẢN (QUAN TRỌNG NHẤT)
def clean_text(text):
    """Hàm xử lý văn bản dùng chung cho cả Train và Predict"""
    if not isinstance(text, str):
        return []
    # 1. Chuyển về chữ thường
    text = text.lower()
    # 2. Loại bỏ ký tự đặc biệt, giữ lại chữ cái và số tiếng Việt
    text = re.sub(r'[^\w\s]', ' ', text)
    # 3. Tách từ đơn giản (Split space)
    words = text.split()
    return words

# 2. MODEL LSTM
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

# 3. LOAD RESOURCE
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

# 4. PREDICT (CÓ TRẢ VỀ DEBUG INFO)
def predict(text, vocab, model):
    if not vocab or not model: return 0.5, [], []
    
    # Dùng hàm clean_text chuẩn
    words = clean_text(text)
    
    # Map sang số
    review_int = [vocab.get(w, 0) for w in words]
    
    # Padding
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

# 5. DATA UTILS
def get_data_files():
    data_path = "data"
    if not os.path.exists(data_path): return []
    return [f for f in os.listdir(data_path) if f.endswith(('.csv', '.xlsx', '.xls'))]

def load_dataset(filename):
    path = os.path.join("data", filename)
    try:
        return pd.read_csv(path) if filename.endswith('.csv') else pd.read_excel(path)
    except: return None

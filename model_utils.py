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

# --- 2. HÀM CHUẨN HÓA TEXT ---
def clean_text(text):
    """Làm sạch văn bản đồng bộ cho cả Train và Test"""
    if not isinstance(text, str): return []
    text = text.lower()
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

# --- 4. CÁC HÀM LOAD DỮ LIỆU ---

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
    """Đọc dữ liệu Training (TXT và CSV)"""
    data_dir = "data"
    all_data = []
    
    train_files = {
        "Negative": "train_negative_tokenized.txt",
        "Neutral": "train_neutral_tokenized.txt",
        "Positive": "train_positive_tokenized.txt"
    }
    
    if os.path.exists(data_dir):
        # 1. Đọc file TXT cũ
        for label, filename in train_files.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            all_data.append({"Content": line.strip(), "Label": label, "Source": filename})

        # 2. Đọc file CSV/Excel mới
        files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx'))]
        for f in files:
            path = os.path.join(data_dir, f)
            try:
                if f.endswith('.csv'): df_temp = pd.read_csv(path)
                else: df_temp = pd.read_excel(path)
                
                # Chuẩn hóa tên cột
                df_temp.rename(columns=lambda x: x.strip(), inplace=True)
                col_mapping = {'Text': 'Content', 'Sentiment': 'Label', 'text': 'Content', 'sentiment': 'Label'}
                df_temp.rename(columns=col_mapping, inplace=True)
                
                if 'Content' in df_temp.columns and 'Label' in df_temp.columns:
                    subset = df_temp[['Content', 'Label']].copy()
                    subset['Label'] = subset['Label'].astype(str).str.strip()
                    subset['Source'] = f
                    all_data.extend(subset.to_dict('records'))
            except Exception as e:
                print(f"Error reading {f}: {e}")

    if not all_data: return pd.DataFrame(columns=['Content', 'Label', 'Source'])
    return pd.DataFrame(all_data)

def load_lexicon_data():
    """Đọc dữ liệu Từ điển (Lexicon) - Bổ sung hàm này"""
    file_path = "data/vietnamese_lexicon.txt"
    if not os.path.exists(file_path): return None
    
    lexicon_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    pos_score = float(parts[2])
                    neg_score = float(parts[3])
                    word = parts[4].split('#')[0].replace('_', ' ') 
                    definition = " ".join(parts[5:]).strip('"')
                    
                    lexicon_data.append({
                        "Từ vựng": word,
                        "Loại từ": parts[0],
                        "Điểm Tích cực": pos_score,
                        "Điểm Tiêu cực": neg_score,
                        "Định nghĩa": definition
                    })
                except ValueError: continue
                    
    if not lexicon_data: return None
    return pd.DataFrame(lexicon_data)

# --- 5. HÀM DỰ ĐOÁN DEBUG ---
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

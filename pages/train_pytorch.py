import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
from collections import Counter

# Import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from model_utils import SentimentLSTM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, clean_text
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

def process_dataframe(df, text_col, label_col):
    df = df.dropna(subset=[text_col, label_col])
    
    # 1. Xử lý Label (QUAN TRỌNG: Quy định rõ ràng)
    y_data = []
    # Nếu label là chuỗi (Negative/Positive)
    if df[label_col].dtype == object:
        y_data = [1 if str(x).lower() in ['pos', 'positive', 'tốt', '1'] else 0 for x in df[label_col]]
    # Nếu label là số (1-5 sao hoặc 0-1)
    else:
        # Giả sử thang 5 sao: >=4 là Tốt (1), <=3 là Xấu (0)
        # Giả sử thang 0-1: >0.5 là Tốt
        y_data = [1 if float(x) >= 4 or (float(x) == 1 and df[label_col].max() == 1) else 0 for x in df[label_col]]

    # 2. Xử lý Text dùng hàm chung
    reviews_cleaned = [clean_text(str(r)) for r in df[text_col]]
    
    # 3. Tạo bộ từ điển (Vocab)
    all_words = [w for sublist in reviews_cleaned for w in sublist]
    count_words = Counter(all_words)
    # Chỉ lấy từ xuất hiện > 1 lần để giảm nhiễu
    sorted_words = [w for w, c in count_words.most_common() if c > 1]
    vocab = {w: i+1 for i, w in enumerate(sorted_words)}
    
    # 4. Map sang số
    reviews_int = []
    for words in reviews_cleaned:
        reviews_int.append([vocab.get(w, 0) for w in words])
        
    # 5. Padding
    seq_len = 50
    features = np.zeros((len(reviews_int), seq_len), dtype=int)
    for i, row in enumerate(reviews_int):
        features[i, -min(len(row), seq_len):] = np.array(row)[:seq_len]
        
    X = torch.from_numpy(features)
    y = torch.from_numpy(np.array(y_data)).float()
    
    return X, y, vocab, None

def show():
    st.title("🔥 Huấn luyện Model (Label Fix)")
    
    if not HAS_DEPS: st.error("Thiếu thư viện."); return

    # Chọn file
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx'))] if os.path.exists(data_dir) else []
    
    if not files: st.warning("Không có file trong data/."); return
    
    col1, col2 = st.columns(2)
    with col1:
        sel_file = st.selectbox("Chọn file:", files)
        path = os.path.join(data_dir, sel_file)
        df = pd.read_csv(path) if sel_file.endswith('.csv') else pd.read_excel(path)
        st.write(f"Đã tải: {len(df)} dòng.")
        
    with col2:
        cols = df.columns.tolist()
        text_col = st.selectbox("Cột nội dung:", cols)
        label_col = st.selectbox("Cột nhãn:", cols)
        
    epochs = st.number_input("Số Epochs:", 1, 50, 10)
    
    if st.button("🚀 Train Lại Từ Đầu"):
        X, y, vocab, err = process_dataframe(df, text_col, label_col)
        
        # Train Loop (Rút gọn)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        vocab_size = len(vocab) + 1
        model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS).to(device)
        criterion = nn.BCELoss(); optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        bar = st.progress(0)
        model.train()
        
        for e in range(epochs):
            h = model.init_hidden(32, device)
            for inp, lbl in loader:
                if inp.size(0) != 32: continue
                h = tuple([each.data for each in h])
                model.zero_grad()
                out, h = model(inp.to(device), h)
                loss = criterion(out, lbl.to(device))
                loss.backward()
                optimizer.step()
            bar.progress((e+1)/epochs)
            
        # Lưu
        if not os.path.exists("models"): os.makedirs("models")
        torch.save(model.state_dict(), "models/sentiment_model.pth")
        with open("models/vocab.pkl", "wb") as f: pickle.dump(vocab, f)
        
        st.success("✅ Train xong! Hãy qua trang Analysis kiểm tra.")

if __name__ == "__main__": show()

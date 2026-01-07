import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
from collections import Counter
import matplotlib.pyplot as plt # Thêm để vẽ biểu đồ đẹp hơn

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

# ... (Giữ nguyên hàm process_dataframe như cũ) ...
def process_dataframe(df, text_col, label_col):
    # Copy lại hàm process_dataframe từ câu trả lời trước
    # (Để ngắn gọn mình không paste lại đoạn xử lý data ở đây, 
    # bạn giữ nguyên logic xử lý data nhé)
    df = df.dropna(subset=[text_col, label_col])
    if df[label_col].dtype == object:
        y_data = [1 if str(x).lower() in ['pos', 'positive', 'tốt', '1'] else 0 for x in df[label_col]]
    else:
        y_data = [1 if float(x) >= 4 or (float(x) == 1 and df[label_col].max() == 1) else 0 for x in df[label_col]]
    
    reviews_cleaned = [clean_text(str(r)) for r in df[text_col]]
    all_words = [w for sublist in reviews_cleaned for w in sublist]
    count_words = Counter(all_words)
    sorted_words = [w for w, c in count_words.most_common() if c > 1]
    vocab = {w: i+1 for i, w in enumerate(sorted_words)}
    
    reviews_int = [[vocab.get(w, 0) for w in words] for words in reviews_cleaned]
    seq_len = 50
    features = np.zeros((len(reviews_int), seq_len), dtype=int)
    for i, row in enumerate(reviews_int):
        features[i, -min(len(row), seq_len):] = np.array(row)[:seq_len]
        
    X = torch.from_numpy(features)
    y = torch.from_numpy(np.array(y_data)).float()
    return X, y, vocab, None

def save_checkpoint(model, vocab, path_model="models/sentiment_model.pth", path_vocab="models/vocab.pkl"):
    """Hàm lưu model an toàn"""
    if not os.path.exists("models"):
        os.makedirs("models")
    
    torch.save(model.state_dict(), path_model)
    with open(path_vocab, "wb") as f:
        pickle.dump(vocab, f)

def show():
    st.title("🔥 Huấn luyện & Theo dõi Log")
    
    # Kiểm tra trạng thái file hiện tại
    if os.path.exists("models/sentiment_model.pth"):
        st.success(f"✅ Đã tìm thấy model cũ tại: `models/sentiment_model.pth`")
    else:
        st.warning("⚠️ Chưa thấy file model. Cần huấn luyện ngay.")

    # ... (Phần chọn file giữ nguyên) ...
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx'))] if os.path.exists(data_dir) else []
    
    if not files: st.warning("Không có file trong data/."); return
    
    col1, col2 = st.columns(2)
    with col1:
        sel_file = st.selectbox("Chọn file:", files)
        path = os.path.join(data_dir, sel_file)
        df = pd.read_csv(path) if sel_file.endswith('.csv') else pd.read_excel(path)
    with col2:
        cols = df.columns.tolist()
        text_col = st.selectbox("Cột nội dung:", cols)
        label_col = st.selectbox("Cột nhãn:", cols)
        
    epochs = st.number_input("Số Epochs:", 1, 100, 5)
    
    col_btn, col_info = st.columns([1, 2])
    
    with col_btn:
        start_train = st.button("🚀 Bắt đầu Train", type="primary")

    if start_train:
        st_status = st.empty()
        st_progress = st.progress(0)
        st_chart = st.empty()
        
        st_status.info("⏳ Đang xử lý dữ liệu...")
        X, y, vocab, err = process_dataframe(df, text_col, label_col)
        
        # Setup Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        vocab_size = len(vocab) + 1
        model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS).to(device)
        criterion = nn.BCELoss(); optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        model.train()
        history = [] # Lưu log loss
        
        for e in range(epochs):
            batch_losses = []
            h = model.init_hidden(32, device)
            
            for inp, lbl in loader:
                if inp.size(0) != 32: continue
                h = tuple([each.data for each in h])
                model.zero_grad()
                out, h = model(inp.to(device), h)
                loss = criterion(out, lbl.to(device))
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            
            # Tính loss trung bình epoch
            avg_loss = np.mean(batch_losses)
            history.append(avg_loss)
            
            # Update UI
            st_status.text(f"Epoch {e+1}/{epochs} - Loss: {avg_loss:.4f}")
            st_progress.progress((e+1)/epochs)
            
            # Vẽ biểu đồ realtime
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(history, marker='o', color='green')
            ax.set_title("Loss History")
            st_chart.pyplot(fig)
            plt.close(fig)

            # === QUAN TRỌNG: LƯU CHECKPOINT SAU MỖI EPOCH ===
            save_checkpoint(model, vocab)
        
        st_status.success("🎉 Hoàn tất! Model đã được lưu vào thư mục `models/`")
        st.balloons()
        
        # Hiển thị file log CSV (giống yêu cầu của bạn)
        log_df = pd.DataFrame({'Epoch': range(1, len(history)+1), 'Loss': history})
        log_df.to_csv("models/history_log.csv", index=False)
        st.dataframe(log_df)

if __name__ == "__main__": show()

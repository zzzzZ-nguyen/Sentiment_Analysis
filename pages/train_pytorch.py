import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
from collections import Counter

# --- CẤU HÌNH ĐƯỜNG DẪN IMPORT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from model_utils import SentimentLSTM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# ==========================================
# 1. HÀM ĐỌC DỮ LIỆU THẬT TỪ FOLDER DATA
# ==========================================
def load_data_from_folder():
    data_path = "data" # Thư mục chứa dữ liệu
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        return None, "Thư mục 'data' không tồn tại. Vui lòng tạo và bỏ file CSV/Excel vào."

    files = [f for f in os.listdir(data_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
    if not files:
        return None, "Không tìm thấy file .csv hoặc .xlsx nào trong thư mục 'data'."
    
    return files, None

def process_dataframe(df, text_col, label_col):
    """Chuyển đổi DataFrame thành format training"""
    # 1. Lọc dữ liệu rỗng
    df = df.dropna(subset=[text_col, label_col])
    
    # 2. Xử lý nhãn (Label) về 0 và 1
    # Logic: Nếu nhãn là số (1-5 sao): >=4 là 1 (Tốt), <=3 là 0 (Tệ)
    # Nếu nhãn là chữ (POS/NEG): 'POS'/'Positive' là 1, còn lại 0
    
    y_data = []
    
    # Kiểm tra kiểu dữ liệu của cột label
    first_val = df[label_col].iloc[0]
    
    try:
        # Trường hợp Label là số (VD: 1,2,3,4,5 hoặc 0,1)
        if isinstance(first_val, (int, float, np.number)):
            # Nếu chỉ có 0 và 1 thì giữ nguyên
            unique_vals = df[label_col].unique()
            if set(unique_vals).issubset({0, 1}):
                y_data = df[label_col].values
            else:
                # Nếu là thang điểm 5 (VD: shopee)
                y_data = [1 if x >= 4 else 0 for x in df[label_col]]
        else:
            # Trường hợp Label là chữ
            y_data = [1 if str(x).lower() in ['pos', 'positive', 'tốt', 'tich cuc', '1'] else 0 for x in df[label_col]]
    except:
        return None, None, None, "Lỗi khi xử lý cột Label. Hãy đảm bảo cột Label chứa số hoặc phân loại rõ ràng."

    # 3. Lấy text
    reviews = df[text_col].astype(str).tolist()
    
    # 4. Tokenize (Tách từ và tạo bộ từ điển)
    # Nối tất cả text lại để đếm từ
    all_text = " ".join(reviews).lower().replace('.', '').replace(',', '')
    words = all_text.split()
    count_words = Counter(words)
    
    # Chỉ giữ lại những từ xuất hiện > 1 lần để giảm nhiễu
    sorted_words = [w for w, c in count_words.most_common() if c > 1]
    vocab = {w: i+1 for i, w in enumerate(sorted_words)}
    
    # Mã hóa reviews thành số
    reviews_int = []
    for r in reviews:
        r_clean = r.lower().replace('.', '').replace(',', '').split()
        reviews_int.append([vocab.get(w, 0) for w in r_clean])
        
    # Padding (Cho bằng độ dài 50)
    seq_len = 50
    features = np.zeros((len(reviews_int), seq_len), dtype=int)
    for i, row in enumerate(reviews_int):
        features[i, -min(len(row), seq_len):] = np.array(row)[:seq_len]
        
    # Convert sang Tensor
    X = torch.from_numpy(features)
    y = torch.from_numpy(np.array(y_data)).float()
    
    return X, y, vocab, None

# ==========================================
# 2. GIAO DIỆN CHÍNH
# ==========================================
def show():
    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("🔥 Train PyTorch với Dữ Liệu Thật")
    st.write("Huấn luyện mô hình từ các file có trong thư mục `data/`.")

    if not HAS_DEPS:
        st.error("⚠️ Thiếu file `model_utils.py` hoặc thư viện `torch`.")
        return

    # --- BƯỚC 1: CHỌN FILE DỮ LIỆU ---
    files, err = load_data_from_folder()
    
    if err:
        st.warning(f"⚠️ {err}")
        st.info("💡 Hãy copy file dữ liệu (CSV hoặc Excel) vào thư mục `data` của dự án.")
        return

    col_file, col_conf = st.columns([1, 2])
    
    with col_file:
        st.subheader("1. Chọn File")
        selected_file = st.selectbox("Chọn file dữ liệu:", files)
        file_path = os.path.join("data", selected_file)
        
        # Đọc file để lấy tên cột
        try:
            if selected_file.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            st.success(f"Đã đọc {len(df)} dòng dữ liệu.")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")
            return

    with col_conf:
        st.subheader("2. Cấu hình Cột")
        all_columns = df.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            text_col = st.selectbox("Cột chứa nội dung (Review):", all_columns, index=0)
        with c2:
            # Cố gắng tự động tìm cột label
            label_index = 0
            for i, col in enumerate(all_columns):
                if col.lower() in ['label', 'rating', 'score', 'sentiment', 'nhãn', 'điểm']:
                    label_index = i
                    break
            label_col = st.selectbox("Cột chứa nhãn (Label/Rating):", all_columns, index=label_index)
            
        st.caption("📝 Ví dụ: Cột nội dung là 'comment', cột nhãn là 'rating' (1-5 sao) hoặc 'label' (0/1).")

    st.write("---")

    # --- BƯỚC 2: TRAIN MODEL ---
    col_train, col_log = st.columns([1, 2])
    
    with col_train:
        st.subheader("3. Huấn luyện")
        epochs = st.number_input("Số vòng lặp (Epochs):", 1, 100, 5)
        batch_size = st.selectbox("Batch Size:", [16, 32, 64], index=1)
        lr = st.select_slider("Learning Rate:", options=[0.01, 0.005, 0.001], value=0.005)
        
        btn_train = st.button("🚀 Bắt đầu Train", type="primary")

    with col_log:
        st.subheader("📈 Tiến trình")
        log_area = st.empty()
        chart_loss = st.empty()
        
        if btn_train:
            status = st.info("🔄 Đang xử lý dữ liệu...")
            
            # Xử lý data thật
            X, y, vocab, err_msg = process_dataframe(df, text_col, label_col)
            
            if err_msg:
                st.error(err_msg)
            else:
                status.info(f"✅ Đã xử lý xong! Vocab: {len(vocab)} từ. Bắt đầu train...")
                time.sleep(1)
                
                # --- CODE TRAIN (GIỐNG CŨ) ---
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Dataset & Loader
                dataset = TensorDataset(X, y)
                train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=False)
                
                # Init Model
                vocab_size = len(vocab) + 1
                model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 1, N_LAYERS)
                model.to(device)
                
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                model.train()
                loss_history = []
                progress_bar = st.progress(0)
                
                start_time = time.time()
                
                for e in range(epochs):
                    h = model.init_hidden(batch_size, device)
                    epoch_losses = []
                    
                    for inputs, labels in train_loader:
                        # Handle batch lẻ
                        curr_bs = inputs.size(0)
                        if curr_bs != batch_size:
                            h = model.init_hidden(curr_bs, device)
                        else:
                            h = tuple([each.data for each in h])
                            
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        model.zero_grad()
                        output, h = model(inputs, h)
                        
                        loss = criterion(output, labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                        
                        epoch_losses.append(loss.item())
                    
                    avg_loss = np.mean(epoch_losses)
                    loss_history.append(avg_loss)
                    
                    # Update Chart & Log
                    chart_loss.line_chart(loss_history)
                    log_area.text(f"Epoch {e+1}/{epochs} | Loss: {avg_loss:.4f}")
                    progress_bar.progress((e + 1) / epochs)
                
                # Lưu Model
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(model.state_dict(), "models/sentiment_model.pth")
                with open("models/vocab.pkl", "wb") as f:
                    pickle.dump(vocab, f)
                
                status.success("🎉 Huấn luyện hoàn tất! Model đã được lưu.")
                st.balloons()
                st.info("👉 Bây giờ bạn có thể qua trang **Analysis** để kiểm tra.")

    st.markdown('</div>', unsafe_allow_html=True)
import time # Import thêm time để sleep

if __name__ == "__main__":
    show()

# ... (Các phần trên giữ nguyên)

def load_training_data_for_app():
    """
    Đọc tất cả dữ liệu từ TXT và CSV trong thư mục data/
    Tự động map cột 'Text' -> 'Content', 'Sentiment' -> 'Label'
    """
    data_dir = "data"
    all_data = []
    
    # 1. ĐỌC CÁC FILE TXT (Cũ)
    train_files = {
        "Negative": "train_negative_tokenized.txt",
        "Neutral": "train_neutral_tokenized.txt",
        "Positive": "train_positive_tokenized.txt"
    }
    
    if os.path.exists(data_dir):
        # Đọc TXT
        for label, filename in train_files.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            all_data.append({"Content": line.strip(), "Label": label, "Source": filename})

        # 2. ĐỌC CÁC FILE CSV/EXCEL (Mới - Bao gồm sentimentdataset.csv)
        files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx'))]
        for f in files:
            path = os.path.join(data_dir, f)
            try:
                # Đọc file
                if f.endswith('.csv'):
                    df_temp = pd.read_csv(path)
                else:
                    df_temp = pd.read_excel(path)
                
                # --- CHUẨN HÓA CỘT ---
                # Đổi tên cột Text/Sentiment thành Content/Label
                df_temp.rename(columns=lambda x: x.strip(), inplace=True) # Xóa khoảng trắng tên cột
                
                # Map tên cột của sentimentdataset.csv sang chuẩn chung
                col_mapping = {
                    'Text': 'Content', 
                    'Sentiment': 'Label',
                    'text': 'Content',
                    'sentiment': 'Label'
                }
                df_temp.rename(columns=col_mapping, inplace=True)
                
                # Kiểm tra xem có đủ 2 cột cần thiết không
                if 'Content' in df_temp.columns and 'Label' in df_temp.columns:
                    # Lọc lấy 2 cột chính
                    subset = df_temp[['Content', 'Label']].copy()
                    
                    # Xử lý sạch nhãn (xóa khoảng trắng thừa: " Positive " -> "Positive")
                    subset['Label'] = subset['Label'].astype(str).str.strip()
                    
                    subset['Source'] = f # Đánh dấu nguồn
                    all_data.extend(subset.to_dict('records'))
            except Exception as e:
                print(f"Lỗi đọc file {f}: {e}")

    # Trả về DataFrame tổng hợp
    if not all_data:
        return pd.DataFrame(columns=['Content', 'Label', 'Source'])
        
    return pd.DataFrame(all_data)

# dump_model_object.py
# ==========================================
# PRO UPGRADE: English Sentiment Analysis Pipeline
# ==========================================

import os
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- CẤU HÌNH ---
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_en.pkl")

def clean_text(text):
    """Hàm làm sạch dữ liệu cơ bản"""
    if not isinstance(text, str): return ""
    text = text.lower()
    # Giữ lại chữ cái tiếng Anh và dấu câu cơ bản, loại bỏ ký tự lạ
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.strip()

def load_real_data():
    """Tự động tìm và load dữ liệu từ file CSV trong thư mục data/"""
    all_texts = []
    all_labels = []
    
    # Dữ liệu mẫu (Backup nếu không tìm thấy file)
    fallback_texts = ["Good job", "Bad quality", "Excellent", "Poor service", "Normal"]
    fallback_labels = ["positive", "negative", "positive", "negative", "neutral"]

    if not os.path.exists(DATA_DIR):
        print("⚠️ Không tìm thấy thư mục data/. Dùng dữ liệu mẫu.")
        return fallback_texts, fallback_labels

    print(f"📂 Đang quét dữ liệu trong {DATA_DIR}...")
    
    # Quét file CSV/Excel
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx'))]
    
    found_data = False
    for f in files:
        # Bỏ qua file metadata
        if "metadata" in f.lower(): continue
        
        path = os.path.join(DATA_DIR, f)
        try:
            df = pd.read_csv(path) if f.endswith('.csv') else pd.read_excel(path)
            
            # Chuẩn hóa tên cột
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Tìm cột text và label phù hợp
            text_col = next((c for c in df.columns if c in ['text', 'content', 'review']), None)
            label_col = next((c for c in df.columns if c in ['sentiment', 'label']), None)
            
            if text_col and label_col:
                print(f"   -> Đọc file: {f} ({len(df)} dòng)")
                # Clean text và thêm vào list
                cleaned_texts = df[text_col].apply(clean_text).tolist()
                labels = df[label_col].astype(str).str.strip().tolist()
                
                all_texts.extend(cleaned_texts)
                all_labels.extend(labels)
                found_data = True
        except Exception as e:
            print(f"   ❌ Lỗi đọc file {f}: {e}")

    if not found_data:
        print("⚠️ Không tìm thấy dữ liệu hợp lệ. Dùng dữ liệu mẫu.")
        return fallback_texts, fallback_labels
    
    return all_texts, all_labels

def train_and_dump():
    # 1. Load Data
    print("\n--- 1. LOAD DATA ---")
    texts, labels = load_real_data()
    print(f"✅ Tổng dữ liệu: {len(texts)} dòng")

    # 2. Split Data (Train 80% - Test 20%)
    # Stratify để đảm bảo tỷ lệ nhãn đều nhau
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        # Nếu dữ liệu quá ít hoặc 1 nhãn chỉ có 1 dòng thì không stratify được
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 3. Tạo Pipeline (QUAN TRỌNG NHẤT)
    # Pipeline giúp gộp vectorizer và model thành 1 khối thống nhất
    print("\n--- 2. TRAINING ---")
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(solver='liblinear', multi_class='auto'))
    ])

    model_pipeline.fit(X_train, y_train)
    print("✅ Training hoàn tất.")

    # 4. Đánh giá Model
    print("\n--- 3. EVALUATION ---")
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Độ chính xác (Accuracy): {acc:.2%}")
    print("\nChi tiết:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5. Lưu Model
    print("\n--- 4. SAVING ---")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Chỉ cần lưu 1 file duy nhất (đã chứa cả vectorizer bên trong)
    joblib.dump(model_pipeline, MODEL_PATH)
    
    print(f"📦 Đã lưu Pipeline vào: {MODEL_PATH}")
    print("💡 Mẹo: Khi dùng, chỉ cần load file này và gọi .predict() trực tiếp với text thô.")

if __name__ == "__main__":
    train_and_dump()

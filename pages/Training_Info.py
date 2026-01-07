import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import collections
import numpy as np

# ==========================================
# 1. CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# ==========================================

# Dữ liệu bạn cung cấp (đã cắt gọn để demo, bạn có thể paste toàn bộ vào đây)
raw_data = """
a	00166146	0.875	0	hấp_dẫn#1	thích nhìn, say mê vẻ đẹp
a	00362467	0.75	0	vui_vẻ#1	một tinh thần tốt, thể hiện tâm trạng rất vui
a	00015589	0.125	0.375	dài#9	có hoặc đang được nhiều hơn mức bình thường hoặc cần thiết
a	00015854	0	0.25	phong_phú#1	có một số lượng lớn
a	00016247	0.125	0.5	thừa_thãi#1	có rất nhiều
a	00065064	0.75	0	tích_cực#3	có tác dụng khẳng định, thúc đẩy sự phát triển
a	00065488	0	0.75	bất_lợi#1	gây thiệt hại
a	00075515	0	0.75	phủ_định#2	bác bỏ sự tồn tại, sự cần thiết của cái gì
a	00220082	0.875	0	xinh_đẹp#1	rất xinh, có được sự hài hòa, trông thích nhìn
a	00193799	0	0.625	khủng_khiếp#1	hoảng sợ hoặc làm cho hoảng sợ ở mức rất cao
a	00422374	0	0.625	tệ_hại#2	quá tệ và có tác dụng gây những tổn thất lớn
a	00328528	0.5	0	mạch_lạc#3	diễn đạt trôi trảy, mạch lạc , từng đoạn một
"""
# Lưu ý: Hãy copy toàn bộ dữ liệu của bạn vào biến raw_data bên trên nếu muốn train hết.

def process_data(raw_text):
    samples = []
    labels = []
    
    lines = raw_text.strip().split('\n')
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 6: continue
        
        # Lấy điểm số và nội dung
        pos_score = float(parts[2])
        neg_score = float(parts[3])
        gloss_text = parts[5] # Sử dụng phần định nghĩa làm đầu vào huấn luyện
        
        # Gán nhãn: 0: Tiêu cực, 1: Tích cực, 2: Trung tính
        if pos_score > neg_score:
            label = 1 
        elif neg_score > pos_score:
            label = 0
        else:
            label = 2 
            
        samples.append(gloss_text.lower()) # Chuyển về chữ thường
        labels.append(label)
    return samples, labels

texts, labels = process_data(raw_data)

# Xây dựng bộ từ điển (Vocabulary)
word_counts = collections.Counter(" ".join(texts).split())
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_idx = {word: i+1 for i, word in enumerate(vocab)} # 0 dành cho padding
word_to_idx['<PAD>'] = 0
vocab_size = len(word_to_idx)

# Hyper-parameters (Đã chỉnh sửa cho phù hợp với Text)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequence_length = 20    # Độ dài tối đa của 1 câu
input_size = 64         # Kích thước vector nhúng (Embedding dim)
hidden_size = 128
num_layers = 2
num_classes = 3         # 3 lớp: Neg, Pos, Neu
batch_size = 5          # Giảm batch size vì dữ liệu ít
num_epochs = 20         # Tăng epoch để máy kịp học
learning_rate = 0.005

# Hàm mã hóa câu văn thành các con số
def encode_text(text, max_len):
    tokens = text.split()
    vec = [word_to_idx.get(token, 0) for token in tokens] # 0 nếu từ không có trong từ điển
    if len(vec) < max_len:
        vec += [0] * (max_len - len(vec)) # Padding
    else:
        vec = vec[:max_len] # Cắt bớt
    return vec

# Dataset Class tùy chỉnh
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text_vec = encode_text(self.texts[idx], sequence_length)
        return torch.tensor(text_vec, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Chia dữ liệu train/test
full_dataset = SentimentDataset(texts, labels)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 2. XÂY DỰNG MODEL (BiRNN cho Text)
# ==========================================

class BiRNN_Text(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(BiRNN_Text, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LỚP MỚI QUAN TRỌNG: Embedding layer chuyển số nguyên thành vector
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes) 
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Chuyển qua lớp Embedding
        # out shape: (batch_size, sequence_length, embed_size)
        out = self.embedding(x)
        
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(out, (h0, c0)) 
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN_Text(vocab_size, input_size, hidden_size, num_layers, num_classes).to(device)

# ==========================================
# 3. TRAINING VÀ TESTING
# ==========================================

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
total_step = len(train_loader)
print("Bắt đầu training...")
for epoch in range(num_epochs):
    for i, (text_vecs, labels) in enumerate(train_loader):
        text_vecs = text_vecs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(text_vecs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model
model.eval() # Chuyển sang chế độ đánh giá (tắt dropout, v.v.)
with torch.no_grad():
    correct = 0
    total = 0
    for text_vecs, labels in test_loader:
        text_vecs = text_vecs.to(device)
        labels = labels.to(device)
        outputs = model(text_vecs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if total > 0:
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    else:
        print("Dataset quá nhỏ để chia train/test, hãy thêm dữ liệu vào biến raw_data.")

# Demo thử nghiệm dự đoán 1 câu
def predict_sentiment(sentence):
    model.eval()
    vec = torch.tensor([encode_text(sentence.lower(), sequence_length)], dtype=torch.long).to(device)
    output = model(vec)
    _, predicted = torch.max(output.data, 1)
    mapping = {0: "Tiêu cực", 1: "Tích cực", 2: "Trung tính"}
    return mapping[predicted.item()]

print("-" * 30)
sample_text = "rất xinh và đáng yêu"
print(f"Dự đoán câu '{sample_text}': {predict_sentiment(sample_text)}")

# Save the model checkpoint
torch.save(model.state_dict(), 'sentiment_model.ckpt')

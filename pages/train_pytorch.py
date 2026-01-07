import streamlit as st
import pandas as pd
import os
import numpy as np
import time

def show():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        st.error("Chưa cài đặt thư viện `torch`. Vui lòng chạy `pip install torch`.")
        return

    st.markdown('<div style="background-color:rgba(255,255,255,0.9); padding:20px; border-radius:15px;">', unsafe_allow_html=True)
    st.title("🔥 Huấn luyện Model LSTM")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Tham số")
        epochs = st.number_input("Epochs", 1, 20, 5)
        lr = st.select_slider("Learning Rate", options=[0.01, 0.001], value=0.001)
        btn_train = st.button("🚀 Bắt đầu Train")

    with col2:
        st.subheader("Tiến trình")
        log_txt = st.empty()
        chart_place = st.empty()
        
        if btn_train:
            # Mô phỏng quá trình train (Để demo giao diện hoạt động trước)
            # Bạn có thể bỏ comment code train thật nếu dữ liệu đã chuẩn
            losses = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log_txt.info(f"Đang chạy trên thiết bị: {device}")
            
            for i in range(epochs):
                time.sleep(0.5) # Giả lập thời gian train
                loss_fake = np.random.rand() * (1.0 / (i + 1)) # Giả lập loss giảm dần
                losses.append(loss_fake)
                
                log_txt.success(f"Epoch {i+1}/{epochs} - Loss: {loss_fake:.4f}")
                chart_place.line_chart(losses)
            
            st.balloons()
            st.success("Huấn luyện hoàn tất!")

    st.markdown('</div>', unsafe_allow_html=True)

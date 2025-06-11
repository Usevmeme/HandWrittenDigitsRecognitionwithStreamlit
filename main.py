import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- モデル定義 ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)           # 入力: 784 -> 出力: 128
        self.bn1 = nn.BatchNorm1d(128)           # バッチ正規化
        self.fc2 = nn.Linear(128, 10)            # 出力: 10クラス

    def forward(self, x):
        x_reshaped = x.view(x.shape[0], -1)      # (batch, 784)
        h = self.fc1(x_reshaped)
        h_bn = self.bn1(h)                       # バッチ正規化
        z = torch.sigmoid(h_bn)                  # 活性化関数
        y_hat = self.fc2(z)
        return y_hat

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルのロード ---
loaded_model = SimpleMLP().to(device)
loaded_model.load_state_dict(torch.load("modelwithBatch.pth", map_location=device, weights_only=True))
loaded_model.eval()

st.title("Digit Classification with SimpleMLP + BatchNorm")
st.write("画像をアップロードして予測を行います。")

# --- 画像アップロード ---
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください（PNG, JPG, JPEG）", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    st.image(image, caption="アップロード画像", use_column_width=False)

    image_np = np.array(image) / 255.0
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = loaded_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    st.write("**予測されたクラス:**", predicted_class)
    st.write("**各クラスの確率:**", probabilities.cpu().numpy())

    fig, ax = plt.subplots()
    ax.imshow(image_tensor.squeeze().cpu().numpy(), cmap="gray")
    ax.set_title(f"Prediction: {predicted_class}")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("画像がアップロードされていません。")

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas


# ------------------------------
# 1. Define your MNIST model (same as the one you trained)
# ------------------------------
from torch import nn
class tinyvgg(nn.Module):
  """"
  tinyVGG replica
  """""
  def __init__(self,input_shape,hidden_shape,output_shape):
    super().__init__()
    self.block1=nn.Sequential(
        nn.Conv2d(input_shape,hidden_shape,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_shape,hidden_shape,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.block2=nn.Sequential(
        nn.Conv2d(hidden_shape,hidden_shape,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_shape,hidden_shape,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_shape*7*7,out_features=output_shape)
    )


  def forward(self,x):
    x=self.block1(x)
    # print(x.shape)
    x=self.block2(x)
    # print(x.shape)
    x=self.classifier(x)
    # print(x.shape)
    return x



# ------------------------------
# 2. Load trained model
# ------------------------------
model = tinyvgg(input_shape=1,hidden_shape=10,output_shape=10)
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.title("🖊️ Handwritten Digit Recognition (MNIST)")

uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess: resize to 28x28, invert, normalize
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Invert if background is white
    image = ImageOps.invert(image)
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    st.subheader(f"🎯 Predicted Digit: **{pred}**")
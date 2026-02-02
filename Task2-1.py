import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load pre-trained ViT
# ---------------------------
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ---------------------------
# Load sample images (CIFAR-10)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# Select 3 images
indices = [0, 1, 2]
images = []
labels = []

for idx in indices:
    img, label = dataset[idx]
    images.append(img)
    labels.append(label)

# ---------------------------
# Prepare inputs using image processor
# ---------------------------
inputs = processor(
    images=[img.permute(1, 2, 0).numpy() for img in images],
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# ---------------------------
# Inference
# ---------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

# ---------------------------
# Display results
# ---------------------------
for i, idx in enumerate(indices):
    predicted_label = model.config.id2label[predictions[i].item()]

    plt.imshow(images[i].permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Top-1 Prediction: {predicted_label}")
    plt.show()

    print(f"Image {i+1}: Top-1 predicted class = {predicted_label}")

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn.functional as F

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load pretrained ViT
# ---------------------------
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True)
model.to(device)
model.eval()

# ---------------------------
# Load a single test image
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

image, label = dataset[0]

# ---------------------------
# Prepare input
# ---------------------------
inputs = processor(
    images=image.permute(1, 2, 0).numpy(),
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# ---------------------------
# Forward pass with attentions
# ---------------------------
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions  # tuple of (num_layers)

# ---------------------------
# Extract last-layer attention
# Shape: (batch, heads, seq_len, seq_len)
# ---------------------------
last_attn = attentions[-1][0]  # remove batch dim → (heads, seq, seq)

# ---------------------------
# Average over heads
# ---------------------------
attn_avg = last_attn.mean(dim=0)  # (seq_len, seq_len)

# ---------------------------
# CLS token attention to patches
# CLS token is at index 0
# ---------------------------
cls_to_patches = attn_avg[0, 1:]  # exclude CLS itself

# ---------------------------
# Reshape to 2D patch grid (14x14)
# ---------------------------
num_patches = cls_to_patches.shape[0]
grid_size = int(np.sqrt(num_patches))

attn_map = cls_to_patches.reshape(grid_size, grid_size)
attn_map = attn_map.cpu().numpy()

# ---------------------------
# Upsample attention map to image resolution
# ---------------------------
attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)
attn_map = F.interpolate(
    attn_map,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)
attn_map = attn_map.squeeze().numpy()

# Normalize
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(6, 6))
plt.imshow(image.permute(1, 2, 0))
plt.imshow(attn_map, cmap="Reds", alpha=0.5)
plt.axis("off")
plt.title("ViT CLS Token Attention Map (Last Layer)")
plt.show()

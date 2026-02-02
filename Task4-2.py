import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from tqdm import tqdm
import random

# ======================================================
# Device
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ======================================================
# Load CLIP
# ======================================================
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ======================================================
# Load STL-10 (test split)
# ======================================================
dataset = STL10(
    root="./data",
    split="test",
    download=True,
    transform=preprocess
)

class_names = dataset.classes

# ======================================================
# Select 50–100 random samples
# ======================================================
num_samples = 100
indices = random.sample(range(len(dataset)), num_samples)
subset = Subset(dataset, indices)

loader = DataLoader(subset, batch_size=32, shuffle=False)

# ======================================================
# Extract image embeddings
# ======================================================
image_embeddings = []
labels = []

with torch.no_grad():
    for images, lbls in tqdm(loader, desc="Extracting image embeddings"):
        images = images.to(device)
        feats = model.encode_image(images)
        image_embeddings.append(feats.cpu())
        labels.extend(lbls.numpy())

image_embeddings = torch.cat(image_embeddings, dim=0).numpy()
labels = np.array(labels)

# ======================================================
# Extract text embeddings (one per label)
# ======================================================
text_prompts = [f"a photo of a {class_names[i]}" for i in labels]
text_tokens = clip.tokenize(text_prompts).to(device)

with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens).cpu().numpy()

# ======================================================
# Optional normalization toggle
# ======================================================
def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

image_embeddings_norm = normalize(image_embeddings)
text_embeddings_norm = normalize(text_embeddings)

# ======================================================
# t-SNE projection
# ======================================================
def tsne_projection(img_emb, txt_emb):
    combined = np.concatenate([img_emb, txt_emb], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    projected = tsne.fit_transform(combined)
    return projected[:num_samples], projected[num_samples:]

# Without normalization
img_tsne, txt_tsne = tsne_projection(image_embeddings, text_embeddings)

# With normalization
img_tsne_norm, txt_tsne_norm = tsne_projection(
    image_embeddings_norm,
    text_embeddings_norm
)

# ======================================================
# Visualization
# ======================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(img_tsne[:, 0], img_tsne[:, 1], c="blue", alpha=0.6, label="Image")
plt.scatter(txt_tsne[:, 0], txt_tsne[:, 1], c="red", alpha=0.6, label="Text")
plt.title("t-SNE (No Normalization)")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(img_tsne_norm[:, 0], img_tsne_norm[:, 1], c="blue", alpha=0.6, label="Image")
plt.scatter(txt_tsne_norm[:, 0], txt_tsne_norm[:, 1], c="red", alpha=0.6, label="Text")
plt.title("t-SNE (With L2 Normalization)")
plt.legend()

plt.tight_layout()
plt.show()

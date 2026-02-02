import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
import random

# ======================================================
# Setup
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ======================================================
# Dataset
# ======================================================
dataset = STL10(
    root="./data",
    split="test",
    download=True,
    transform=preprocess
)

class_names = dataset.classes

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
    for images, lbls in tqdm(loader, desc="Image embeddings"):
        images = images.to(device)
        feats = model.encode_image(images)
        image_embeddings.append(feats.cpu())
        labels.extend(lbls.numpy())

image_embeddings = torch.cat(image_embeddings).numpy()
labels = np.array(labels)

# ======================================================
# Extract paired text embeddings
# ======================================================
text_prompts = [f"a photo of a {class_names[i]}" for i in labels]
text_tokens = clip.tokenize(text_prompts).to(device)

with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens).cpu().numpy()

# ======================================================
# Normalize (important for CLIP)
# ======================================================
def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

X = normalize(image_embeddings)
Y = normalize(text_embeddings)

# ======================================================
# Orthogonal Procrustes Alignment
# ======================================================
R, _ = orthogonal_procrustes(X, Y)

X_aligned = X @ R

# ======================================================
# t-SNE Visualization
# ======================================================
def plot_tsne(A, B, title):
    combined = np.vstack([A, B])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(combined)

    n = A.shape[0]
    plt.scatter(proj[:n, 0], proj[:n, 1], c="blue", alpha=0.6, label="Image")
    plt.scatter(proj[n:, 0], proj[n:, 1], c="red", alpha=0.6, label="Text")
    plt.legend()
    plt.title(title)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_tsne(X, Y, "Before Procrustes Alignment")

plt.subplot(1, 2, 2)
plot_tsne(X_aligned, Y, "After Procrustes Alignment")

plt.tight_layout()
plt.show()

# ======================================================
# Zero-shot Classification Accuracy
# ======================================================
def zero_shot_accuracy(image_feats, text_feats, labels):
    sims = image_feats @ text_feats.T
    preds = sims.argmax(axis=1)
    return (preds == np.arange(len(labels))).mean()

acc_before = zero_shot_accuracy(X, Y, labels)
acc_after = zero_shot_accuracy(X_aligned, Y, labels)

print(f"Accuracy before alignment: {acc_before:.4f}")
print(f"Accuracy after alignment:  {acc_after:.4f}")

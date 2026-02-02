import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Data transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# CIFAR-10 (subset for visualization)
# ---------------------------
dataset = datasets.CIFAR10(
    root="./data", train=False, transform=transform, download=True
)

subset_size = 2000  # t-SNE is expensive
dataset = torch.utils.data.Subset(dataset, range(subset_size))

loader = DataLoader(dataset, batch_size=64, shuffle=False)

# ---------------------------
# Load pretrained ResNet-152
# ---------------------------
model = models.resnet152(pretrained=True)
model.eval()
model = model.to(device)

# ---------------------------
# Feature extractor
# ---------------------------
features = {"early": [], "middle": [], "late": []}
labels = []

def hook_fn(name):
    def hook(module, input, output):
        pooled = torch.mean(output, dim=[2, 3])  # Global Avg Pool
        features[name].append(pooled.detach().cpu())
    return hook

# Register hooks
model.layer1.register_forward_hook(hook_fn("early"))
model.layer3.register_forward_hook(hook_fn("middle"))
model.layer4.register_forward_hook(hook_fn("late"))

# ---------------------------
# Forward pass
# ---------------------------
with torch.no_grad():
    for images, targets in loader:
        images = images.to(device)
        _ = model(images)
        labels.append(targets)

labels = torch.cat(labels).numpy()

# Concatenate features
for key in features:
    features[key] = torch.cat(features[key]).numpy()

# ---------------------------
# t-SNE visualization
# ---------------------------
def plot_tsne(feats, labels, title):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(feats)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
        s=10
    )
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1))
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_tsne(features["early"], labels, "Early Layer Features (layer1)")
plot_tsne(features["middle"], labels, "Middle Layer Features (layer3)")
plot_tsne(features["late"], labels, "Late Layer Features (layer4)")

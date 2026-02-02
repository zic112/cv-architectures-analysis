import torch
import clip
import numpy as np
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ======================================================
# Device
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ======================================================
# Load CLIP model (official OpenAI implementation)
# ======================================================
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ======================================================
# STL-10 Dataset
# ======================================================
test_dataset = STL10(
    root="./data",
    split="test",
    download=True,
    transform=preprocess
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

class_names = test_dataset.classes
num_classes = len(class_names)

# ======================================================
# Prompting strategies
# ======================================================
prompts_plain = [name for name in class_names]

prompts_photo = [f"a photo of a {name}" for name in class_names]

prompts_descriptive = [
    f"a clear photo of a {name}, a type of object in the real world"
    for name in class_names
]

prompt_sets = {
    "Plain labels": prompts_plain,
    "Photo prompt": prompts_photo,
    "Descriptive prompt": prompts_descriptive
}

# ======================================================
# Zero-shot classifier creation
# ======================================================
def build_text_features(prompts):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

# ======================================================
# Evaluation
# ======================================================
def zeroshot_accuracy(text_features):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100

# ======================================================
# Run experiments
# ======================================================
results = {}

for name, prompts in prompt_sets.items():
    print(f"\nEvaluating: {name}")
    text_features = build_text_features(prompts)
    acc = zeroshot_accuracy(text_features)
    results[name] = acc
    print(f"Accuracy: {acc:.2f}%")

# ======================================================
# Summary
# ======================================================
print("\n==== Zero-Shot Accuracy on STL-10 ====")
for k, v in results.items():
    print(f"{k}: {v:.2f}%")

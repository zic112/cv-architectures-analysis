import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load ViT
# ---------------------------
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ---------------------------
# Dataset
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

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ---------------------------
# Patch masking utilities
# ---------------------------
PATCH_SIZE = 16
GRID_SIZE = 224 // PATCH_SIZE  # 14x14 patches

def mask_random_patches(images, mask_ratio):
    """
    Randomly mask a fraction of patches.
    """
    images = images.clone()
    B, C, H, W = images.shape
    num_patches = GRID_SIZE * GRID_SIZE
    num_mask = int(mask_ratio * num_patches)

    for b in range(B):
        patch_indices = np.random.choice(num_patches, num_mask, replace=False)
        for idx in patch_indices:
            i = idx // GRID_SIZE
            j = idx % GRID_SIZE
            images[b, :, i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                          j*PATCH_SIZE:(j+1)*PATCH_SIZE] = 0.0
    return images

def mask_center_patches(images, mask_ratio):
    """
    Mask a centered square region of patches.
    """
    images = images.clone()
    num_mask_side = int(np.sqrt(mask_ratio) * GRID_SIZE)
    start = (GRID_SIZE - num_mask_side) // 2
    end = start + num_mask_side

    for i in range(start, end):
        for j in range(start, end):
            images[:, :, i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                          j*PATCH_SIZE:(j+1)*PATCH_SIZE] = 0.0
    return images

# ---------------------------
# Evaluation function
# ---------------------------
def evaluate(mask_fn=None, mask_ratio=0.0):
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if mask_fn is not None:
                images = mask_fn(images, mask_ratio)

            inputs = processor(
                images=images.permute(0, 2, 3, 1).cpu().numpy(),
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

# ---------------------------
# Run experiments
# ---------------------------
mask_ratios = [0.0, 0.1, 0.3, 0.5]

print("Mask Ratio | Random Mask Acc | Center Mask Acc")
for r in mask_ratios:
    acc_rand = evaluate(mask_random_patches, r)
    acc_center = evaluate(mask_center_patches, r)
    print(f"{r:.1f}       | {acc_rand:.2f}%           | {acc_center:.2f}%")

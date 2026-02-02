import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load pretrained ViT backbone (no classifier head)
# ---------------------------
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
# Note: Load report warnings (Unexpected/Missing) are normal when loading just the backbone
vit = ViTModel.from_pretrained(model_name)
vit.to(device)
vit.eval()

for param in vit.parameters():
    param.requires_grad = False

# ---------------------------
# Dataset
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10("./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

NUM_CLASSES = 10
HIDDEN_DIM = vit.config.hidden_size

# ---------------------------
# Linear probe models
# ---------------------------
class LinearProbe(nn.Module):
    def __init__(self, mode="cls"):
        super().__init__()
        self.mode = mode
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, outputs):
        # outputs.last_hidden_state: (B, seq_len, D)
        if self.mode == "cls":
            x = outputs.last_hidden_state[:, 0]          # CLS token
        elif self.mode == "mean":
            x = outputs.last_hidden_state[:, 1:].mean(1) # mean of patch tokens
        return self.fc(x)

# ---------------------------
# Training & evaluation
# ---------------------------
def train_and_eval(mode):
    probe = LinearProbe(mode).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)

    # Train
    for epoch in range(5):
        probe.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Added do_rescale=False to resolve the warning
            inputs = processor(
                images=images,
                return_tensors="pt",
                do_rescale=False
            ).to(device)

            with torch.no_grad():
                outputs = vit(**inputs)

            logits = probe(outputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            inputs = processor(
                images=images,
                return_tensors="pt",
                do_rescale=False
            ).to(device)

            outputs = vit(**inputs)
            preds = probe(outputs).argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

# ---------------------------
# Run comparison
# ---------------------------
acc_cls = train_and_eval("cls")
acc_mean = train_and_eval("mean")

print(f"CLS token linear probe accuracy:  {acc_cls:.2f}%")
print(f"Mean patch token accuracy:        {acc_mean:.2f}%")
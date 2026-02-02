import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Hyperparameters
# ---------------------------
batch_size = 64
num_epochs = 5
learning_rate = 1e-3
num_classes = 10

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
# CIFAR-10
# ---------------------------
train_dataset = datasets.CIFAR10(root="./data", train=True,
                                 transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False,
                                transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# Load ResNet-152
# ---------------------------
model = models.resnet152(pretrained=True)

# ---------------------------
# Disable skip connections in selected blocks
# ---------------------------
def disable_skip_connections(resnet_model, layer_name="layer4", block_indices=[-1]):
    """
    Disable skip connections by removing identity mapping
    in selected residual blocks.
    """
    layer = getattr(resnet_model, layer_name)
    for idx in block_indices:
        block = layer[idx]
        block.downsample = None

        def forward_no_skip(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out = self.relu(out)
            return out

        block.forward = forward_no_skip.__get__(block, block.__class__)

# Disable skip connection in the last block of layer4
disable_skip_connections(model, layer_name="layer4", block_indices=[-1])

# ---------------------------
# Replace classifier
# ---------------------------
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# ---------------------------
# Loss & Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# ---------------------------
# Training / Evaluation
# ---------------------------
def train_one_epoch(model, loader):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, pred = torch.max(out, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    return loss_sum / len(loader), 100 * correct / total


def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return loss_sum / len(loader), 100 * correct / total

# ---------------------------
# Run training
# ---------------------------
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, test_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

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
num_epochs = 10
learning_rate = 1e-4
num_classes = 10

# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
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
# Model factory
# ---------------------------
def get_model(pretrained=True):
    model = models.resnet152(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# ---------------------------
# Train / Eval
# ---------------------------
def train_and_eval(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100 * correct / total

# ---------------------------
# Run experiments
# ---------------------------
acc_pretrained = train_and_eval(get_model(pretrained=True))
acc_scratch = train_and_eval(get_model(pretrained=False))

print(f"Pretrained accuracy: {acc_pretrained:.2f}%")
print(f"From scratch accuracy: {acc_scratch:.2f}%")

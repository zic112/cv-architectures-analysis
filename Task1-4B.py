import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
num_epochs = 10
learning_rate = 1e-4
num_classes = 10

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10("./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def get_model(finetune_mode="last"):
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    if finetune_mode == "last":
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif finetune_mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    return model.to(device)

def train_and_eval(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate)

    for _ in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100 * correct / total

acc_last = train_and_eval(get_model("last"))
acc_full = train_and_eval(get_model("full"))

print(f"Fine-tune last block accuracy: {acc_last:.2f}%")
print(f"Fine-tune full backbone accuracy: {acc_full:.2f}%")

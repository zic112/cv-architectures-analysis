import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ======================================================
# Device
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================================================
# Hyperparameters
# ======================================================
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
LATENT_DIM = 20
KL_WARMUP_EPOCHS = 10   # KL annealing duration

# ======================================================
# Dataset
# ======================================================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    "./data", train=True, download=True, transform=transform
)
val_dataset = datasets.FashionMNIST(
    "./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================
# VAE Architecture (UNCHANGED)
# ======================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ======================================================
# Model, Optimizer, Loss
# ======================================================
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
mse_loss = nn.MSELoss(reduction="sum")

def vae_loss(recon_x, x, mu, logvar, beta):
    recon = mse_loss(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

# ======================================================
# Training with KL Annealing
# ======================================================
train_losses, kl_values = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    total_kl = 0

    # Linear KL warm-up
    beta = min(1.0, epoch / KL_WARMUP_EPOCHS)

    for x, _ in train_loader:
        x = x.to(device)

        recon_x, mu, logvar = model(x)
        loss, _, kl = vae_loss(recon_x, x, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kl += kl.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_kl = total_kl / len(train_loader.dataset)

    train_losses.append(avg_loss)
    kl_values.append(avg_kl)

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"Beta={beta:.2f} "
        f"Loss={avg_loss:.4f} "
        f"KL={avg_kl:.4f}"
    )

# ======================================================
# Save Model
# ======================================================
torch.save(model.state_dict(), "vae_kl_annealed.pth")

# ======================================================
# Plot KL progression
# ======================================================
plt.plot(kl_values)
plt.xlabel("Epoch")
plt.ylabel("KL Divergence")
plt.title("KL Divergence During Training (KL Annealing)")
plt.show()

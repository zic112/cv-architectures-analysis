import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================================================
# Device
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================================================
# Hyperparameters
# ======================================================
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
LATENT_DIM = 20

# ======================================================
# Dataset: FashionMNIST
# ======================================================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

val_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================
# VAE Architecture
# ======================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # Decoder
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
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ======================================================
# Model, Optimizer, Loss
# ======================================================
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
reconstruction_loss_fn = nn.MSELoss(reduction="sum")

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = reconstruction_loss_fn(recon_x, x)
    kl_div = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return recon_loss + kl_div, recon_loss, kl_div

# ======================================================
# Training Loop
# ======================================================
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):

    # ----------------------
    # Training
    # ----------------------
    model.train()
    train_loss = 0.0

    for x, _ in train_loader:
        x = x.to(device)

        recon_x, mu, logvar = model(x)
        loss, _, _ = vae_loss(recon_x, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # ----------------------
    # Validation
    # ----------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon_x, x, mu, logvar)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

# ======================================================
# Save Model
# ======================================================
torch.save(model.state_dict(), "vae_fashionmnist.pth")

print("\nTraining complete.")
print("Model saved as vae_fashionmnist.pth")
print("\nTraining losses:", train_losses)
print("Validation losses:", val_losses)

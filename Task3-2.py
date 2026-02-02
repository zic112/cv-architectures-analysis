import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================================================
# Device
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Hyperparameters (must match training)
# ======================================================
LATENT_DIM = 20
BATCH_SIZE = 16

# ======================================================
# Dataset
# ======================================================
transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ======================================================
# VAE Architecture (same as training)
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
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ======================================================
# Load trained model
# ======================================================
model = VAE().to(device)
model.load_state_dict(torch.load("vae_fashionmnist.pth", map_location=device))
model.eval()

# ======================================================
# Utility: plot image grids
# ======================================================
def plot_images(images, title):
    images = images.cpu()
    plt.figure(figsize=(8, 2))
    for i in range(images.size(0)):
        plt.subplot(1, images.size(0), i + 1)
        plt.imshow(images[i, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# ======================================================
# (a) Reconstructions
# ======================================================
x, _ = next(iter(test_loader))
x = x.to(device)

with torch.no_grad():
    recon_x, _, _ = model(x)

plot_images(x, "Original Images")
plot_images(recon_x, "VAE Reconstructions")

# ======================================================
# (b) Generations from Gaussian prior
# ======================================================
with torch.no_grad():
    z = torch.randn(BATCH_SIZE, LATENT_DIM).to(device)
    samples_gaussian = model.decode(z)

plot_images(samples_gaussian, "Generated Samples (Gaussian Prior)")

# ======================================================
# (c) Generations from Laplacian prior
# ======================================================
with torch.no_grad():
    laplace_dist = torch.distributions.Laplace(0, 1)
    z_laplace = laplace_dist.sample((BATCH_SIZE, LATENT_DIM)).to(device)
    samples_laplace = model.decode(z_laplace)

plot_images(samples_laplace, "Generated Samples (Laplacian Prior)")

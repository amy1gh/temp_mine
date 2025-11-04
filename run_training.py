import os
import argparse
import torch
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from disvae.training import Trainer
from disvae.models.vae import init_specific_model
from utils.helpers import get_device, set_seed
from disvae.models.losses import BetaHLoss

loss_f = BetaHLoss(beta=4, rec_dist='bernoulli', steps_anneal=1000)


# ------------------------------
# Config
# ------------------------------
parser = argparse.ArgumentParser(description="Train VAE on MNIST or CIFAR-10")
parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist",
                    help="Dataset to train on")
parser.add_argument("--latent", type=int, default=6, help="Latent dimension")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--res-dir", type=str, default="results_mine", help="Results directory")
args = parser.parse_args()

SEED = args.seed
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LATENT_DIM = args.latent
DATASET = args.dataset
MODEL_NAME = f"btcvae_{DATASET}_{LATENT_DIM}"
RES_DIR = args.res_dir
DEVICE = get_device(is_gpu=True)
SAVE_DIR = os.path.join(RES_DIR, MODEL_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

set_seed(SEED)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Training Device: {DEVICE}")
logger.info(f"Dataset: {DATASET}, Latent dim: {LATENT_DIM}, Save: {SAVE_DIR}")

# ------------------------------
# Dataset
# ------------------------------
if DATASET == "mnist":
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    img_size = (1, 32, 32)
elif DATASET == "cifar10":
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    img_size = (3, 32, 32)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
# Model
# ------------------------------
model = init_specific_model(model_type="Burgess",
                            img_size=img_size,
                            latent_dim=LATENT_DIM)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss function
# loss_f = model.loss_function  # directly from the model


# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  loss_f=loss_f,
                  device=DEVICE,
                  logger=logger,
                  save_dir=SAVE_DIR,
                  is_progress_bar=True)

# ------------------------------
# Run training
# ------------------------------
trainer(train_loader, epochs=EPOCHS, checkpoint_every=5)
logger.info(f"Training completed. Results saved to {SAVE_DIR}")

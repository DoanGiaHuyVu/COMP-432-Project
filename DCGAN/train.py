import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import gc
from model import Discriminator, Generator, initialize_weights
from torchvision.models import inception_v3
import os
import multiprocessing
from torchvision.utils import save_image
import torch.nn.functional as F


# Inception model for FID calculation
class InceptionModel(nn.Module):
    def __init__(self):
        super(InceptionModel, self).__init__()
        self.model = inception_v3(pretrained=True)
        self.model.fc = nn.Identity()  # Remove final fully connected layer
        self.model.eval()

    def forward(self, x):
        # Resize images to inception input size
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features before the final classification layer
        x = self.model(x)
        return x


# Inception model for IS calculation (uses softmax output)
class InceptionModelIS(nn.Module):
    def __init__(self):
        super(InceptionModelIS, self).__init__()
        self.model = inception_v3(pretrained=True)
        self.model.eval()

    def forward(self, x):
        # Resize images to inception input size
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # We want the logits before softmax
        x = self.model(x)
        # If we get a tuple (Some versions of inception return logits and aux_logits)
        if isinstance(x, tuple):
            x = x[0]  # Get logits
        return x


# Function to calculate covariance matrix
def torch_cov(m):
    m_centered = m - m.mean(dim=0)
    factor = 1 / (m.shape[0] - 1)
    return factor * m_centered.t() @ m_centered


# Function to extract features using inception model
def get_inception_features(images, inception_model, device, batch_size=32):
    features = []

    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        with torch.no_grad():
            batch_features = inception_model(batch).cpu()
        features.append(batch_features)

    return torch.cat(features, dim=0)


# Function to calculate Inception Score
def calculate_inception_score(images, inception_model, device, batch_size=32, splits=10):
    # Get predictions
    preds = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(device)
            batch_preds = F.softmax(inception_model(batch), dim=1).cpu().numpy()
            preds.append(batch_preds)

    preds = np.concatenate(preds, axis=0)

    # Calculate mean and standard deviation of Inception Score
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits):(i + 1) * (len(preds) // splits), :]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


# Function to calculate FID score
def calculate_fid(real_features, fake_features):
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(dim=0), torch_cov(real_features)
    mu2, sigma2 = fake_features.mean(dim=0), torch_cov(fake_features)

    # Calculate squared difference between means
    diff = mu1 - mu2

    # Calculate FID score
    covmean, _ = linalg.sqrtm(sigma1.cpu().numpy() @ sigma2.cpu().numpy(), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1.cpu().numpy() + sigma2.cpu().numpy() - 2 * covmean)
    return fid


def calculate_accuracy(disc_real, disc_fake):
    # For real images, higher scores are better (should be close to 1)
    real_accuracy = (disc_real > 0.5).float().mean()
    # For fake images, lower scores are better (should be close to 0)
    fake_accuracy = (disc_fake < 0.5).float().mean()
    # Overall accuracy is the average
    accuracy = (real_accuracy + fake_accuracy) / 2
    return accuracy.item()


# Function to plot metrics
def plot_metrics(gen_losses, disc_losses, accuracies, fid_scores, is_scores=None, save_path="graphs"):
    os.makedirs(save_path, exist_ok=True)

    # Plot generator and discriminator losses
    plt.figure(figsize=(10, 6))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses')
    plt.legend()
    plt.savefig(f"{save_path}/losses.png")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Discriminator Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Discriminator Accuracy')
    plt.legend()
    plt.savefig(f"{save_path}/accuracy.png")
    plt.close()

    # Plot FID scores
    if fid_scores and len(fid_scores) > 0:  # Only plot if there are scores
        plt.figure(figsize=(10, 6))
        plt.plot(fid_scores, label='FID Score')
        plt.xlabel('Epochs')
        plt.ylabel('FID Score')
        plt.title('FID Score (lower is better)')
        plt.legend()
        plt.savefig(f"{save_path}/fid_scores.png")
        plt.close()

    # Plot Inception Scores
    if is_scores and len(is_scores) > 0:  # Only plot if there are scores
        plt.figure(figsize=(10, 6))
        means = [score[0] for score in is_scores]
        stds = [score[1] for score in is_scores]
        epochs = list(range(len(is_scores)))

        plt.errorbar(epochs, means, yerr=stds, fmt='-o', label='Inception Score')
        plt.xlabel('Epochs')
        plt.ylabel('Inception Score')
        plt.title('Inception Score (higher is better)')
        plt.legend()
        plt.savefig(f"{save_path}/inception_scores.png")
        plt.close()


# Hyperparameters etc.
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
LEARNING_RATE = 1e-5
LEARNING_RATE_G = 2e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3  # put 1 for MNIST
NOISE_DIM = 100
NUM_EPOCHS = 100  # Increased from 1 to 5 to better track metrics over time
FEATURES_DISC = 64
FEATURES_GEN = 64
FID_FREQUENCY = 1  # Calculate FID every N epochs
IS_FREQUENCY = 1  # Calculate IS every N epochs
CHECKPOINT_FREQ = 10  # Save checkpoints every N epochs
SAVE_DIR = "checkpoints"

# Create directories for saving
os.makedirs('generated_images', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Transforms
transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Ensure consistent image size
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# For tensorboard plotting
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

# Dataset
dataset = datasets.ImageFolder(root="../swing_sequence", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Get a fixed set of real images for FID calculation
real_samples_for_fid = []
try:
    sample_dataloader = DataLoader(dataset, batch_size=2000, shuffle=True)
    for data, _ in sample_dataloader:
        real_samples_for_fid = data
        break  # Just need one batch
except Exception as e:
    print(f"Warning: Could not load samples for FID: {e}")
    # Continue without FID calculation if it fails

# Load inception models for FID and IS calculation
inception_model = None
inception_model_is = None
try:
    inception_model = InceptionModel().to(device)
    for param in inception_model.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"Warning: Could not load inception model for FID: {e}")
    print("FID calculation will be skipped.")

try:
    inception_model_is = InceptionModelIS().to(device)
    for param in inception_model_is.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"Warning: Could not load inception model for IS: {e}")
    print("Inception Score calculation will be skipped.")


def save_checkpoint(state, filename):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file, model, optimizer):
    print(f"=> Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


# Initialize models
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

# Optimizers and Loss
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Tracking metrics
gen_losses = []
disc_losses = []
accuracies = []
fid_scores = []
is_scores = []

# Fixed noise for consistent generation
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0

# Training
gen.train()
disc.train()

try:
    for epoch in range(NUM_EPOCHS):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        epoch_accuracy = 0
        batches_counted = 0

        # Progress bar for better tracking
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (real, _) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                # Get batch size and move to device
                real = real.to(device)
                current_batch_size = real.shape[0]

                # Create labels
                real_label = torch.ones(current_batch_size, 1).to(device)
                fake_label = torch.zeros(current_batch_size, 1).to(device)

                # Generate fake images
                noise = torch.randn(current_batch_size, NOISE_DIM, 1, 1).to(device)
                fake = gen(noise)

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                # Real images
                disc_real = disc(real).reshape(-1, 1)
                loss_disc_real = criterion(disc_real, real_label)

                # Fake images
                disc_fake = disc(fake.detach()).reshape(-1, 1)
                loss_disc_fake = criterion(disc_fake, fake_label)

                # Combined loss
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

                # Optimize discriminator
                disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # Generate fake images again (no detach this time)
                output = disc(fake).reshape(-1, 1)
                loss_gen = criterion(output, real_label)  # We want generator to produce images that fool discriminator

                # Optimize generator
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                # Calculate accuracy
                accuracy = calculate_accuracy(disc_real.detach(), disc_fake.detach())

                # Accumulate metrics
                epoch_gen_loss += loss_gen.item()
                epoch_disc_loss += loss_disc.item()
                epoch_accuracy += accuracy
                batches_counted += 1

                # Update progress bar
                tepoch.set_postfix(
                    D_loss=f"{loss_disc.item():.4f}",
                    G_loss=f"{loss_gen.item():.4f}",
                    Accuracy=f"{accuracy:.4f}"
                )

                # Print info and save images periodically
                if batch_idx % 100 == 0:
                    # Log to tensorboard
                    writer_fake.add_scalar("Loss/Generator", loss_gen.item(), global_step=step)
                    writer_real.add_scalar("Loss/Discriminator", loss_disc.item(), global_step=step)
                    writer_real.add_scalar("Metrics/Accuracy", accuracy, global_step=step)

                    # Generate and save sample images
                    with torch.no_grad():
                        fake_samples = gen(fixed_noise)
                        # Take real samples for comparison
                        real_samples = real[:32] if real.size(0) >= 32 else real

                        # Create grid of images
                        img_grid_real = torchvision.utils.make_grid(real_samples, normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake_samples, normalize=True)

                        # Add to tensorboard
                        writer_real.add_image("Real", img_grid_real, global_step=step)
                        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                        # Save image grid
                        save_image(
                            fake_samples,
                            f"generated_images/fake_grid_epoch{epoch}_batch{batch_idx}.png",
                            nrow=8,
                            normalize=True
                        )

                    step += 1

                # Clean up memory
                del real, fake, noise, disc_real, disc_fake, output
                if device == "cuda":
                    torch.cuda.empty_cache()
                else:
                    gc.collect()

        # Calculate average losses and accuracy for the epoch
        avg_gen_loss = epoch_gen_loss / batches_counted
        avg_disc_loss = epoch_disc_loss / batches_counted
        avg_accuracy = epoch_accuracy / batches_counted

        # Store epoch metrics
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        accuracies.append(avg_accuracy)

        print(
            f"Epoch {epoch} - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # Save a grid of generated images after each epoch
        with torch.no_grad():
            fake_samples = gen(fixed_noise)
            save_image(
                fake_samples,
                f"generated_images/epoch{epoch}_samples.png",
                nrow=8,
                normalize=True
            )

        # Calculate FID score if inception model is available
        if inception_model is not None and len(real_samples_for_fid) > 0 and (
                epoch % FID_FREQUENCY == 0 or epoch == NUM_EPOCHS - 1):
            try:
                print("Calculating FID score...")
                with torch.no_grad():
                    # Generate fake images for FID calculation
                    n_samples = min(2000, len(real_samples_for_fid))
                    fake_samples_for_fid = []
                    for i in range(0, n_samples, BATCH_SIZE):
                        batch_size = min(BATCH_SIZE, n_samples - i)
                        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
                        fake = gen(noise).cpu()
                        fake_samples_for_fid.append(fake)
                        del noise, fake

                    fake_samples_for_fid = torch.cat(fake_samples_for_fid, dim=0)[:n_samples]

                    # Extract features using inception model
                    real_features = get_inception_features(real_samples_for_fid[:n_samples], inception_model, device)
                    fake_features = get_inception_features(fake_samples_for_fid, inception_model, device)

                    # Calculate FID score
                    fid = calculate_fid(real_features, fake_features)
                    fid_scores.append(fid)

                    print(f"FID Score at epoch {epoch}: {fid}")
                    writer_fake.add_scalar("Metrics/FID", fid, global_step=epoch)

                    # Clean up
                    del real_features, fake_features, fake_samples_for_fid
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    else:
                        gc.collect()
            except Exception as fid_error:
                print(f"Warning: FID calculation failed: {fid_error}")

        # Calculate Inception Score if model is available
        if inception_model_is is not None and (epoch % IS_FREQUENCY == 0 or epoch == NUM_EPOCHS - 1):
            try:
                print("Calculating Inception Score...")
                with torch.no_grad():
                    # Generate fake images for IS calculation
                    n_samples = 1000  # Number of samples to use for IS
                    fake_samples_for_is = []
                    for i in range(0, n_samples, BATCH_SIZE):
                        batch_size = min(BATCH_SIZE, n_samples - i)
                        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
                        fake = gen(noise).cpu()
                        fake_samples_for_is.append(fake)
                        del noise, fake

                    fake_samples_for_is = torch.cat(fake_samples_for_is, dim=0)[:n_samples]

                    # Calculate Inception Score
                    is_mean, is_std = calculate_inception_score(fake_samples_for_is, inception_model_is, device)
                    is_scores.append((is_mean, is_std))

                    print(f"Inception Score at epoch {epoch}: {is_mean} ± {is_std}")
                    writer_fake.add_scalar("Metrics/IS", is_mean, global_step=epoch)

                    # Clean up
                    del fake_samples_for_is
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    else:
                        gc.collect()
            except Exception as is_error:
                print(f"Warning: Inception Score calculation failed: {is_error}")

        # Save checkpoint
        if epoch % CHECKPOINT_FREQ == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_gen = {
                "state_dict": gen.state_dict(),
                "optimizer": opt_gen.state_dict(),
            }
            checkpoint_disc = {
                "state_dict": disc.state_dict(),
                "optimizer": opt_disc.state_dict(),
            }
            save_checkpoint(checkpoint_gen, os.path.join(SAVE_DIR, f"gen_epoch{epoch}.pth.tar"))
            save_checkpoint(checkpoint_disc, os.path.join(SAVE_DIR, f"disc_epoch{epoch}.pth.tar"))

        # Plot metrics after each epoch
        plot_metrics(gen_losses, disc_losses, accuracies, fid_scores, is_scores)

except Exception as e:
    print(f"Training interrupted with error: {e}")
    # Save checkpoint on error
    checkpoint_gen = {
        "state_dict": gen.state_dict(),
        "optimizer": opt_gen.state_dict(),
    }
    checkpoint_disc = {
        "state_dict": disc.state_dict(),
        "optimizer": opt_disc.state_dict(),
    }
    save_checkpoint(checkpoint_gen, os.path.join(SAVE_DIR, "gen_interrupted.pth.tar"))
    save_checkpoint(checkpoint_disc, os.path.join(SAVE_DIR, "disc_interrupted.pth.tar"))

finally:
    # Close tensorboard writers
    writer_real.close()
    writer_fake.close()
    # Final plotting of metrics
    plot_metrics(gen_losses, disc_losses, accuracies, fid_scores, is_scores)

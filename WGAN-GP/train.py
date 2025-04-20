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
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights
from torchvision.models import inception_v3
import os
import multiprocessing
from torchvision.utils import save_image
import torch.nn.functional as F

# Fix multiprocessing issue on macOS
if __name__ == "__main__":
    # On macOS, we need to use the 'spawn' start method
    multiprocessing.set_start_method('spawn', force=True)


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


# Function to calculate accuracy (using discriminator confidence)
def calculate_accuracy(critic_real, critic_fake):
    # For real images, higher scores are better
    real_accuracy = (critic_real > 0).float().mean()
    # For fake images, lower scores are better
    fake_accuracy = (critic_fake < 0).float().mean()
    # Overall accuracy is the average
    accuracy = (real_accuracy + fake_accuracy) / 2
    return accuracy.item()


# Function to plot metrics
def plot_metrics(gen_losses, critic_losses, accuracies, fid_scores, is_scores=None, save_path="plots"):
    os.makedirs(save_path, exist_ok=True)

    # Plot generator and critic losses
    plt.figure(figsize=(10, 6))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Steps (hundreds)')
    plt.ylabel('Loss')
    plt.title('Generator and Critic Losses')
    plt.legend()
    plt.savefig(f"{save_path}/losses.png")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Discriminator Accuracy')
    plt.xlabel('Steps (hundreds)')
    plt.ylabel('Accuracy')
    plt.title('Discriminator Accuracy')
    plt.legend()
    plt.savefig(f"{save_path}/accuracy.png")
    plt.close()

    # Plot FID scores
    if fid_scores:  # Only plot if there are scores
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
        epochs = list(range(0, len(is_scores) * IS_FREQUENCY, IS_FREQUENCY))

        plt.errorbar(epochs, means, yerr=stds, fmt='-o', label='Inception Score')
        plt.xlabel('Epochs')
        plt.ylabel('Inception Score')
        plt.title('Inception Score (higher is better)')
        plt.legend()
        plt.savefig(f"{save_path}/inception_scores.png")
        plt.close()


# Hyperparameters etc.
#[I 2025-04-18 00:24:26,089] Trial 3 finished with value: -0.7979182518752027 and parameters: {'lr_gen': 1.2270077346608413e-05, 'lr_critic': 2.6242904115202502e-06, 'batch_size': 64, 'z_dim': 189, 'features_gen': 44, 'features_critic': 17, 'critic_iterations': 4, 'lambda_gp': 14.492435773543217, 'beta1': 0.8718747770848536, 'beta2': 0.9645054468315092}. Best is trial 2 with value: -0.999217101232739.
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
LEARNING_RATE = 2.6242904115202502e-06  #1.2719455210012978e-06
LEARNING_RATE_G = 1.2270077346608413e-05  #6.377024347464574e-06
BATCH_SIZE = 64  # Reduced batch size to prevent memory issues
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 189  #218
NUM_EPOCHS = 100
FEATURES_CRITIC = 17  #41
FEATURES_GEN = 44  #33
CRITIC_ITERATIONS = 4  #3
LAMBDA_GP = 14.492435773543217  #16.31574404190011
FID_FREQUENCY = 1  # Calculate FID every N epochs
IS_FREQUENCY = 1  # Calculate IS every N epochs
CHECKPOINT_FREQ = 10  # Save checkpoints every N epochs
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.ImageFolder(root="../swing_sequence", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,  # Reduced number of workers to prevent potential issues
    pin_memory=True if device != "cpu" else False,
)

# Get a fixed set of real images for FID calculation
real_samples_for_fid = []
try:
    sample_dataloader = DataLoader(dataset, batch_size=5000, shuffle=True)
    for data, _ in sample_dataloader:
        real_samples_for_fid = data
        break  # Just need one batch
except Exception as e:
    print(f"Warning: Could not load samples for FID: {e}")
    # Continue without FID calculation if it fails

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# Try to load the inception models, but continue if they fail
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

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G,
                     betas=(0.8718747770848536, 0.9645054468315092))  # 0.6283091138832372, 0.9530799188832499
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.8718747770848536, 0.9645054468315092))

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

# Create directory for saving generated images
GENERATED_IMAGES_DIR = "generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# Lists to store metrics
gen_losses = []
critic_losses = []
accuracies = []
fid_scores = []
is_scores = []

gen.train()
critic.train()

try:
    for epoch in range(NUM_EPOCHS):
        epoch_gen_loss = 0
        epoch_critic_loss = 0
        epoch_accuracy = 0
        batches_counted = 0

        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # Calculate accuracy
                accuracy = calculate_accuracy(critic_real, critic_fake)
                epoch_accuracy += accuracy

                # Clear from memory to prevent OOM
                del gp

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Add to epoch totals
            epoch_gen_loss += loss_gen.item()
            epoch_critic_loss += loss_critic.item()
            batches_counted += 1

            # Clean up memory
            del fake, noise, critic_real, critic_fake, gen_fake, loss_gen, loss_critic
            torch.cuda.empty_cache() if device == "cuda" else gc.collect()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                real_for_viz = real.detach().clone()[:32]
                cur_gen_loss = epoch_gen_loss / batches_counted
                cur_critic_loss = epoch_critic_loss / batches_counted
                cur_accuracy = epoch_accuracy / (batches_counted * CRITIC_ITERATIONS)

                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {cur_critic_loss:.4f}, loss G: {cur_gen_loss:.4f}, Accuracy: {cur_accuracy:.4f}"
                )

                # Store metrics
                gen_losses.append(cur_gen_loss)
                critic_losses.append(cur_critic_loss)
                accuracies.append(cur_accuracy)

                with torch.no_grad():
                    try:
                        fake = gen(fixed_noise)
                        # take out (up to) 32 examples
                        real_samples = real_for_viz[:32] if len(real_for_viz) >= 32 else real_for_viz
                        img_grid_real = torchvision.utils.make_grid(real_samples, normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                        writer_real.add_image("Real", img_grid_real, global_step=step)
                        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                        # Also log metrics to tensorboard
                        writer_fake.add_scalar("Loss/Generator", cur_gen_loss, global_step=step)
                        writer_real.add_scalar("Loss/Critic", cur_critic_loss, global_step=step)
                        writer_real.add_scalar("Metrics/Accuracy", cur_accuracy, global_step=step)

                    except Exception as viz_error:
                        print(f"Warning: Visualization failed: {viz_error}")

                del real_for_viz
                step += 1

                # Clear memory
                if 'fake' in locals():
                    del fake
                if 'img_grid_real' in locals():
                    del img_grid_real
                if 'img_grid_fake' in locals():
                    del img_grid_fake
                torch.cuda.empty_cache() if device == "cuda" else gc.collect()

            # Make sure to properly delete the 'real' variable at the end of each batch
            del real

        # Calculate average losses and accuracy for the epoch
        avg_gen_loss = epoch_gen_loss / batches_counted
        avg_critic_loss = epoch_critic_loss / batches_counted
        avg_accuracy = epoch_accuracy / (batches_counted * CRITIC_ITERATIONS)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] complete - Avg Gen Loss: {avg_gen_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

        # Save generated images after each epoch
        print(f"Saving generated images for epoch {epoch}...")
        with torch.no_grad():
            # Generate a larger grid for visualization (8x8 = 64 images)
            visualization_noise = torch.randn(64, Z_DIM, 1, 1).to(device)
            fake_samples = gen(visualization_noise)

            # Save as a grid
            save_image(
                fake_samples,
                os.path.join(GENERATED_IMAGES_DIR, f"fake_samples_epoch_{epoch}.png"),
                normalize=True,
                nrow=8  # 8 images per row for an 8x8 grid
            )

            # Also save individual images for closer inspection
            # Create epoch-specific subfolder
            epoch_dir = os.path.join(GENERATED_IMAGES_DIR, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save a subset of individual images (first 8)
            for img_idx in range(min(8, fake_samples.size(0))):
                save_image(
                    fake_samples[img_idx],
                    os.path.join(epoch_dir, f"sample_{img_idx}.png"),
                    normalize=True
                )

            # Clean up to avoid memory issues
            del fake_samples, visualization_noise
            torch.cuda.empty_cache() if device == "cuda" else gc.collect()

        # Calculate FID score every few epochs, but only if inception model is available
        if inception_model is not None and len(real_samples_for_fid) > 0 and (
                epoch % FID_FREQUENCY == 0 or epoch == NUM_EPOCHS - 1):
            try:
                print("Calculating FID score...")
                with torch.no_grad():
                    # Generate fake images for FID calculation
                    n_samples = min(2000, len(real_samples_for_fid))  # Reduced sample count for memory
                    fake_samples_for_fid = []
                    for i in range(0, n_samples, BATCH_SIZE):
                        batch_size = min(BATCH_SIZE, n_samples - i)
                        noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
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
                    torch.cuda.empty_cache() if device == "cuda" else gc.collect()
            except Exception as fid_error:
                print(f"Warning: FID calculation failed: {fid_error}")

        # Calculate Inception Score every few epochs, but only if inception model is available
        if inception_model_is is not None and (epoch % IS_FREQUENCY == 0 or epoch == NUM_EPOCHS - 1):
            try:
                print("Calculating Inception Score...")
                with torch.no_grad():
                    # Generate fake images for IS calculation
                    n_samples = 1000  # Number of samples to use for IS
                    fake_samples_for_is = []
                    for i in range(0, n_samples, BATCH_SIZE):
                        batch_size = min(BATCH_SIZE, n_samples - i)
                        noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
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
                    torch.cuda.empty_cache() if device == "cuda" else gc.collect()
            except Exception as is_error:
                print(f"Warning: Inception Score calculation failed: {is_error}")

        # Save checkpoint - Match the existing save_checkpoint format from utils.py
        if epoch % CHECKPOINT_FREQ == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_gen = {
                'gen': gen.state_dict(),
                'disc': None  # Not used but keeping structure consistent
            }
            checkpoint_critic = {
                'gen': None,  # Not used but keeping structure consistent
                'disc': critic.state_dict()
            }

            save_checkpoint(checkpoint_gen, filename=os.path.join(SAVE_DIR, f"gen_{epoch}.pth.tar"))
            save_checkpoint(checkpoint_critic, filename=os.path.join(SAVE_DIR, f"critic_{epoch}.pth.tar"))

        # Plot metrics after each epoch to track progress even if training stops early
        plot_metrics(gen_losses, critic_losses, accuracies, fid_scores, is_scores)

except Exception as e:
    print(f"Training interrupted with error: {e}")
    # Save checkpoint on error, properly formatted to match utils.py function
    checkpoint_gen = {
        'gen': gen.state_dict(),
        'disc': None
    }
    checkpoint_critic = {
        'gen': None,
        'disc': critic.state_dict()
    }

    save_checkpoint(checkpoint_gen, filename=os.path.join(SAVE_DIR, "gen_interrupted.pth.tar"))
    save_checkpoint(checkpoint_critic, filename=os.path.join(SAVE_DIR, "critic_interrupted.pth.tar"))

    # Plot metrics
    plot_metrics(gen_losses, critic_losses, accuracies, fid_scores, is_scores)

finally:
    # Close tensorboard writers
    writer_real.close()
    writer_fake.close()
    # Final plotting of metrics
    plot_metrics(gen_losses, critic_losses, accuracies, fid_scores, is_scores)
    print("Training completed or interrupted!")

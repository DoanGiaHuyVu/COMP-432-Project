import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
from utils import gradient_penalty
from model import Discriminator, Generator, initialize_weights
import os
import multiprocessing
import optuna
import joblib

# Fix multiprocessing issue on macOS
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)


# Function to calculate accuracy (using discriminator confidence)
def calculate_accuracy(critic_real, critic_fake):
    real_accuracy = (critic_real > 0).float().mean()
    fake_accuracy = (critic_fake < 0).float().mean()
    accuracy = (real_accuracy + fake_accuracy) / 2
    return accuracy.item()


# Define Optuna objective function for hyperparameter tuning
def objective(trial):
    # Define hyperparameters to tune
    lr_gen = trial.suggest_float("lr_gen", 1e-6, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    z_dim = trial.suggest_int("z_dim", 64, 256)
    features_gen = trial.suggest_int("features_gen", 8, 64)
    features_critic = trial.suggest_int("features_critic", 8, 64)
    critic_iterations = trial.suggest_int("critic_iterations", 1, 5)
    lambda_gp = trial.suggest_float("lambda_gp", 5.0, 20.0)
    beta1 = trial.suggest_float("beta1", 0.0, 0.9)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)

    # Define constant hyperparameters
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    NUM_EPOCHS = 3  # Reduced for quicker trials

    # Setup data transformations
    transforms_list = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    dataset = datasets.ImageFolder(root="../swing_sequence", transform=transforms_list)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device != "cpu" else False,
    )

    # Initialize models
    gen = Generator(z_dim, CHANNELS_IMG, features_gen).to(device)
    critic = Discriminator(CHANNELS_IMG, features_critic).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # Initialize optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(beta1, beta2))
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic, betas=(beta1, beta2))

    # Lists to track metrics
    accuracies = []
    gen.train()
    critic.train()
    step = 0

    # Train for specified number of epochs
    for epoch in range(NUM_EPOCHS):
        epoch_accuracy = 0
        batches_counted = 0

        for batch_idx, (real, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic
            for _ in range(critic_iterations):
                noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # Calculate accuracy
                accuracy = calculate_accuracy(critic_real, critic_fake)
                epoch_accuracy += accuracy

            # Train Generator
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            batches_counted += 1

            # Clean up memory
            del fake, noise, critic_real, critic_fake, gen_fake, loss_gen, loss_critic, gp
            torch.cuda.empty_cache() if device == "cuda" else gc.collect()

            # Report progress less frequently
            if batch_idx % 200 == 0 and batch_idx > 0:
                cur_accuracy = epoch_accuracy / (batches_counted * critic_iterations)
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} Accuracy: {cur_accuracy:.4f}")

                # Store metrics
                accuracies.append(cur_accuracy)

                # Report intermediate values to Optuna
                trial.report(cur_accuracy, step)

                # Check if we should prune the trial
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                step += 1

            # Delete real at the end of each batch
            del real

    # Return the mean of the last 10 accuracy values as optimization target
    return -np.mean(accuracies[-10:])  # Negative because we want to maximize accuracy


def main():
    # Create study directory
    STUDY_DIR = "optuna_study"
    os.makedirs(STUDY_DIR, exist_ok=True)

    # Create a new study object
    print("Creating Optuna study for GAN hyperparameter tuning...")
    study = optuna.create_study(
        direction="minimize",  # We're minimizing negative accuracy (maximizing accuracy)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name="gan_hyperparameter_tuning"
    )

    # Run the optimization
    print("Starting hyperparameter tuning...")
    study.optimize(objective, n_trials=20, timeout=None)

    # Print the best hyperparameters
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {-trial.value}")  # Convert back to positive accuracy
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the study
    joblib.dump(study, f"{STUDY_DIR}/study.pkl")

    # Save best hyperparameters
    with open(f"{STUDY_DIR}/best_hyperparameters.txt", "w") as f:
        f.write(f"Best accuracy: {-trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

    print("Hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
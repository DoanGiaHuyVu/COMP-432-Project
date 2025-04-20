import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import gradient_penalty
from model import Discriminator, Generator, initialize_weights
import optuna
from optuna.exceptions import TrialPruned
import joblib


def objective(trial):
    # ─── Hyperparameters ────────────────────────────────────────────────────────
    lr_gen = trial.suggest_float("lr_gen", 1e-6, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    z_dim = trial.suggest_int("z_dim", 64, 256)
    f_gen = trial.suggest_int("features_gen", 8, 64)
    f_critic = trial.suggest_int("features_critic", 8, 64)
    critic_iters = trial.suggest_int("critic_iterations", 1, 5)
    lambda_gp = trial.suggest_float("lambda_gp", 5.0, 20.0)
    beta1 = trial.suggest_float("beta1", 0.0, 0.9)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available()
    else "cpu")

    # ─── DataLoader ─────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(root="../swing_sequence", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2,
                        pin_memory=(device != "cpu"))

    # ─── Models & Optims ─────────────────────────────────────────────────────────
    gen = Generator(z_dim, 3, f_gen).to(device)
    critic = Discriminator(3, f_critic).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    opt_g = optim.Adam(gen.parameters(), lr=lr_gen, betas=(beta1, beta2))
    opt_c = optim.Adam(critic.parameters(), lr=lr_critic, betas=(beta1, beta2))

    # ─── Training Loop ───────────────────────────────────────────────────────────
    NUM_EPOCHS = 3
    for epoch in range(NUM_EPOCHS):
        for real, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            real = real.to(device)
            bsz = real.size(0)

            # Critic updates
            for _ in range(critic_iters):
                noise = torch.randn(bsz, z_dim, 1, 1, device=device)
                fake = gen(noise)

                cr = critic(real).view(-1)
                cf = critic(fake).view(-1)
                gp = gradient_penalty(critic, real, fake, device=device)

                loss_c = -(cr.mean() - cf.mean()) + lambda_gp * gp
                critic.zero_grad()
                loss_c.backward(retain_graph=True)
                opt_c.step()

            # Generator update
            noise = torch.randn(bsz, z_dim, 1, 1, device=device)
            fake = gen(noise)
            gf = critic(fake).view(-1)
            loss_g = -gf.mean()
            gen.zero_grad()
            loss_g.backward()
            opt_g.step()

            # cleanup
            del real, fake, cr, cf, gf, loss_c, loss_g, gp
            if device == "cuda":
                torch.cuda.empty_cache()
            else:
                gc.collect()

        # ─── Intermediate W‑distance & Pruning Check ───────────────────────────────
        real_batch, _ = next(iter(loader))
        real_batch = real_batch.to(device)
        with torch.no_grad():
            z = torch.randn(real_batch.size(0), z_dim, 1, 1, device=device)
            fake_batch = gen(z)

        wr = critic(real_batch).view(-1).mean().item()
        wf = critic(fake_batch).view(-1).mean().item()
        w_dist = wr - wf

        # report to Optuna and maybe prune
        trial.report(w_dist, epoch)
        if trial.should_prune():
            raise TrialPruned()

    # ─── Final return ────────────────────────────────────────────────────────────
    return w_dist


def main():
    STUDY_DIR = "optuna_study"
    os.makedirs(STUDY_DIR, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",  # we want the smallest W‑distance
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="wgan_gp_wasserstein_tuning"
    )
    study.optimize(objective, n_trials=20)

    # Save results…
    best = study.best_trial
    print(f"Lowest W-distance: {best.value:.4f}")
    print("Hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    txt_path = os.path.join(STUDY_DIR, "best_hyperparameters.txt")
    with open(txt_path, "w") as f:
        f.write(f"Lowest Wasserstein distance: {best.value:.6f}\n\n")
        f.write("Hyperparameters:\n")
        for name, val in best.params.items():
            f.write(f"  {name}: {val}\n")
    joblib.dump(study, os.path.join(STUDY_DIR, "study.pkl"))
    print(f"Best hyperparameters saved to {txt_path}")


if __name__ == "__main__":
    main()

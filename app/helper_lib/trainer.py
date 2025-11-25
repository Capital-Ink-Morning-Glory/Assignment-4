import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm


@torch.no_grad()
def clip_img(x):
    return torch.clamp((x + 1) / 2, 0, 1)  # Scale from [-1,1] to [0,1]


def plot_samples(samples, n=8):
    samples = clip_img(samples)
    samples = samples.cpu()
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        img = samples[i].permute(1, 2, 0).squeeze()  # CHW to HWC
        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].axis("off")
    plt.show()


def train_ebm(model, train_loader, test_loader, optimizer, device, epochs):
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model

    # Training loop
    for epoch in range(epochs):
        model.reset_metrics()
        for index, batch in enumerate(train_loader):
            real_imgs = batch[0].to(device)
            metrics = model.train_step(real_imgs, optimizer)

        plot_samples(torch.cat(model.buffer.examples[-8:]), n=8)
        print(f"Epoch {epoch + 1} - " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        # Validation step
        model.reset_metrics()
        for batch in test_loader:
            real_imgs = batch[0].to(device)
            val_metrics = model.test_step(real_imgs)

        print(f"Validation - " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
    return model


def train_diffusion(model, train_loader, val_loader, optimizer, loss_fn, epochs, device, checkpoint_dir):
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []
        loader_with_progress = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for images, _ in loader_with_progress:
            images = images.to(device)
            loss = model.train_step(images, optimizer, loss_fn)
            train_losses.append(loss)
            loader_with_progress.set_postfix(loss=f'{loss:.4f}')

        avg_train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        model.plot_images()

        val_losses = []
        for images, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
            images = images.to(device)
            loss = model.test_step(images, loss_fn)
            val_losses.append(loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        loader_with_progress.set_postfix(loss=f'{avg_train_loss:.4f}')

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.network.state_dict(),
            'ema_model_state_dict': model.ema_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'normalizer_mean': model.normalizer_mean,
            'normalizer_std': model.normalizer_std
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch + 1:03d}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'diffusion_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f"New best model saved at epoch {epoch + 1} with val_loss: {avg_val_loss:.4f}")

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load a saved checkpoint and restore model, EMA, and optimizer states
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore normalizer settings
    model.normalizer_mean = checkpoint['normalizer_mean']
    model.normalizer_std = checkpoint['normalizer_std']

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")

    return checkpoint['epoch']

import copy
import math
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from app.helper_lib.generator import generate_ebm_samples


def get_model(model_name):
    # TODO: define and return the appropriate model_name - one of: EBM and Diffusion.
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model_name == 'EBM':  # Energy Based Models
        # Swish activation function
        def swish(x):
            return x * torch.sigmoid(x)

        class EnergyModel(nn.Module):
            def __init__(self):
                super(EnergyModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
                self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(64 * 2 * 2, 64)
                self.fc2 = nn.Linear(64, 1)

            def forward(self, x):
                x = swish(self.conv1(x))
                x = swish(self.conv2(x))
                x = swish(self.conv3(x))
                x = swish(self.conv4(x))
                x = self.flatten(x)
                x = swish(self.fc1(x))
                return self.fc2(x)

        class Buffer:
            def __init__(self, model, device):
                super().__init__()
                self.model = model
                self.device = device
                # Start with random images in the buffer
                self.examples = [torch.rand((1, 3, 32, 32), device=self.device) * 2 - 1 for _ in range(128)]

            def sample_new_exmps(self, steps, step_size, noise):
                n_new = np.random.binomial(128, 0.05)

                # Generate new random images for around 5% of the inputs
                new_rand_imgs = torch.rand((n_new, 3, 32, 32), device=self.device) * 2 - 1

                # Sample old images from the buffer for the rest
                old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)

                inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

                # Run Langevin dynamics
                new_imgs = generate_ebm_samples(self.model, inp_imgs, steps, step_size, noise)

                # Update buffer
                self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
                self.examples = self.examples[:8192]

                return new_imgs

        class Metric:
            def __init__(self):
                self.reset()

            def update(self, val):
                self.total += val.item()
                self.count += 1

            def result(self):
                return self.total / self.count if self.count > 0 else 0.0

            def reset(self):
                self.total = 0.0
                self.count = 0

        class EBM(nn.Module):
            def __init__(self, model, alpha, steps, step_size, noise, device):
                super().__init__()
                self.device = device
                # Define the nn energy model
                self.model = model

                self.buffer = Buffer(self.model, device=device)

                # Define the hyperparameters
                self.alpha = alpha
                self.steps = steps
                self.step_size = step_size
                self.noise = noise

                self.loss_metric = Metric()
                self.reg_loss_metric = Metric()
                self.cdiv_loss_metric = Metric()
                self.real_out_metric = Metric()
                self.fake_out_metric = Metric()

            def metrics(self):
                return {
                    "loss": self.loss_metric.result(),
                    "reg": self.reg_loss_metric.result(),
                    "cdiv": self.cdiv_loss_metric.result(),
                    "real": self.real_out_metric.result(),
                    "fake": self.fake_out_metric.result()
                }

            def reset_metrics(self):
                for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric, self.real_out_metric,
                          self.fake_out_metric]:
                    m.reset()

            def train_step(self, real_imgs, optimizer):
                real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
                real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

                fake_imgs = self.buffer.sample_new_exmps(steps=self.steps, step_size=self.step_size, noise=self.noise)

                inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
                inp_imgs = inp_imgs.clone().detach().to(device).requires_grad_(False)

                out_scores = self.model(inp_imgs)

                real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)],
                                                 dim=0)

                cdiv_loss = real_out.mean() - fake_out.mean()
                reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
                loss = cdiv_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

                optimizer.step()

                self.loss_metric.update(loss)
                self.reg_loss_metric.update(reg_loss)
                self.cdiv_loss_metric.update(cdiv_loss)
                self.real_out_metric.update(real_out.mean())
                self.fake_out_metric.update(fake_out.mean())

                return self.metrics()

            def test_step(self, real_imgs):
                batch_size = real_imgs.shape[0]
                fake_imgs = torch.rand((batch_size, 3, 32, 32), device=self.device) * 2 - 1

                inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

                with torch.no_grad():
                    out_scores = self.model(inp_imgs)
                    real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
                    cdiv = real_out.mean() - fake_out.mean()

                self.cdiv_loss_metric.update(cdiv)
                self.real_out_metric.update(real_out.mean())
                self.fake_out_metric.update(fake_out.mean())

                return {
                    "cdiv": self.cdiv_loss_metric.result(),
                    "real": self.real_out_metric.result(),
                    "fake": self.fake_out_metric.result()
                }

        nn_energy_model = EnergyModel()
        nn_energy_model.to(device)
        model = EBM(nn_energy_model, alpha=0.1, steps=60, step_size=10, noise=0.005, device=device)

        return nn_energy_model, model

    elif model_name == "Diffusion":
        def show_image(image, label=None):
            print("Label: ", label)
            plt.figure(figsize=(4, 3))
            plt.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')
            plt.show()

        # Intuition: Similar to cosine schedule but avoids extreme values.
        def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
            # Flatten diffusion_times to handle any shape
            original_shape = diffusion_times.shape
            diffusion_times_flat = diffusion_times.flatten()

            # Compute start and end angles from signal rate bounds
            start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
            end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

            # Linearly interpolate angles
            diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

            # Compute signal and noise rates
            signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
            noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

            return noise_rates, signal_rates

        class SinusoidalEmbedding(nn.Module):
            def __init__(self, num_frequencies=16):
                super().__init__()
                self.num_frequencies = num_frequencies
                frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
                self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

            def forward(self, x):
                """
                x: Tensor of shape (B, 1, 1, 1)
                returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
                """
                x = x.expand(-1, 1, 1, self.num_frequencies)
                sin_part = torch.sin(self.angular_speeds * x)
                cos_part = torch.cos(self.angular_speeds * x)
                return torch.cat([sin_part, cos_part], dim=-1)

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.needs_projection = (in_channels != out_channels)
                if self.needs_projection:
                    self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                else:
                    self.proj = nn.Identity()

                self.norm = nn.BatchNorm2d(in_channels, affine=False)
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            def swish(self, x):
                return x * torch.sigmoid(x)

            def forward(self, x):
                residual = self.proj(x)
                # x = self.norm(x)
                x = self.swish(self.conv1(x))
                x = self.conv2(x)
                return x + residual

        class DownBlock(nn.Module):
            def __init__(self, width, block_depth, in_channels):
                super().__init__()
                self.blocks = nn.ModuleList()
                for i in range(block_depth):
                    self.blocks.append(ResidualBlock(in_channels, width))
                    in_channels = width
                self.pool = nn.AvgPool2d(kernel_size=2)

            def forward(self, x, skips):
                for block in self.blocks:
                    x = block(x)
                    skips.append(x)
                x = self.pool(x)
                return x

        class UpBlock(nn.Module):
            def __init__(self, width, block_depth, in_channels):
                super().__init__()
                self.blocks = nn.ModuleList()
                for _ in range(block_depth):
                    self.blocks.append(ResidualBlock(in_channels + width, width))
                    in_channels = width

            def forward(self, x, skips):
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                for block in self.blocks:
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = block(x)
                return x

        class UNet(nn.Module):
            def __init__(self, image_size, num_channels, embedding_dim=32):
                super().__init__()
                self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
                self.num_channels = num_channels
                self.image_size = image_size
                self.embedding_dim = embedding_dim
                self.embedding = SinusoidalEmbedding(num_frequencies=16)
                self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

                self.down1 = DownBlock(32, in_channels=64, block_depth=2)
                self.down2 = DownBlock(64, in_channels=32, block_depth=2)
                self.down3 = DownBlock(96, in_channels=64, block_depth=2)

                self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
                self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

                self.up1 = UpBlock(96, in_channels=128, block_depth=2)
                self.up2 = UpBlock(64, block_depth=2, in_channels=96)
                self.up3 = UpBlock(32, block_depth=2, in_channels=64)

                self.final = nn.Conv2d(32, num_channels, kernel_size=1)
                nn.init.zeros_(self.final.weight)  # Keep zero init like TF reference

            def forward(self, noisy_images, noise_variances):
                skips = []
                x = self.initial(noisy_images)
                noise_emb = self.embedding(noise_variances)  # Shape: (B, 1, 1, 32)
                # Upsample to match image size like TF reference
                noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size),
                                          mode='nearest')
                x = torch.cat([x, noise_emb], dim=1)

                x = self.down1(x, skips)
                x = self.down2(x, skips)
                x = self.down3(x, skips)

                x = self.mid1(x)
                x = self.mid2(x)

                x = self.up1(x, skips)
                x = self.up2(x, skips)
                x = self.up3(x, skips)

                return self.final(x)

        class DiffusionModel(nn.Module):
            def __init__(self, model, schedule_fn):
                super().__init__()
                self.network = model
                self.ema_network = copy.deepcopy(model)
                self.ema_network.eval()
                self.ema_decay = 0.8
                self.schedule_fn = schedule_fn
                self.normalizer_mean = 0.0
                self.normalizer_std = 1.0

            def to(self, device):
                # Override to() to ensure both networks move to the same device
                super().to(device)
                self.ema_network.to(device)
                return self

            def set_normalizer(self, mean, std):
                self.normalizer_mean = mean
                self.normalizer_std = std

            def denormalize(self, x):
                return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

            def denoise(self, noisy_images, noise_rates, signal_rates, training):
                # Use EMA network for inference, main network for training
                if training:
                    network = self.network
                    network.train()
                else:
                    network = self.ema_network
                    network.eval()

                pred_noises = network(noisy_images, noise_rates ** 2)
                pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
                return pred_noises, pred_images

            def reverse_diffusion(self, initial_noise, diffusion_steps):
                step_size = 1.0 / diffusion_steps
                current_images = initial_noise
                for step in range(diffusion_steps):
                    t = (torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) *
                         (1 - step * step_size))
                    noise_rates, signal_rates = self.schedule_fn(t)
                    pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

                    # Debug generation process
                    if step % max(1, diffusion_steps // 4) == 0:  # Print 4 times during generation
                        print(f"Generation Step {step}/{diffusion_steps}: t={1 - step * step_size:.3f}")
                        print(f"  Current images std: {current_images.std().item():.4f}")
                        print(f"  Pred images std: {pred_images.std().item():.4f}")
                        print(
                            f"  Signal rate: {signal_rates.mean().item():.4f}, Noise rate: {noise_rates.mean().item():.4f}")

                    next_diffusion_times = t - step_size
                    next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
                    current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
                return pred_images

            def generate(self, num_images, diffusion_steps, image_size=32, initial_noise=None):
                if initial_noise is None:
                    initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size),
                                                device=next(self.parameters()).device)
                with torch.no_grad():
                    return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

            def train_step(self, images, optimizer, loss_fn):
                images = (images - self.normalizer_mean) / self.normalizer_std
                noises = torch.randn_like(images)

                diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
                noise_rates, signal_rates = self.schedule_fn(diffusion_times)
                noisy_images = signal_rates * images + noise_rates * noises

                pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
                loss = loss_fn(pred_noises, noises)

                # Debug prints
                if torch.rand(1).item() < 0.01:  # Print more frequently to see output
                    print(
                        f"Debug - Loss: {loss.item():.4f}, Noise std: {noises.std().item():.4f}, Pred std: {pred_noises.std().item():.4f}")
                    print(f"Signal rates range: {signal_rates.min().item():.4f}-{signal_rates.max().item():.4f}")
                    print(f"Noise rates range: {noise_rates.min().item():.4f}-{noise_rates.max().item():.4f}")

                optimizer.zero_grad()
                loss.backward()

                # Check for gradient issues
                if torch.rand(1).item() < 0.01:
                    total_norm = 0
                    for p in self.network.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    print(f"Gradient norm: {total_norm:.4f}")

                optimizer.step()

                with torch.no_grad():
                    # Debug EMA update occasionally
                    if torch.rand(1).item() < 0.001:
                        param_diff = 0
                        for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                            param_diff += (ema_param - param).abs().mean().item()
                        print(f"EMA Update Debug - Avg param difference: {param_diff:.6f}")

                    for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                        ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)

                return loss.item()

            def test_step(self, images, loss_fn):
                images = (images - self.normalizer_mean) / self.normalizer_std
                noises = torch.randn_like(images)

                diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
                noise_rates, signal_rates = self.schedule_fn(diffusion_times)
                noisy_images = signal_rates * images + noise_rates * noises

                with torch.no_grad():
                    pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
                    loss = loss_fn(pred_noises, noises)

                return loss.item()

            def plot_images(self, num_rows=3, num_cols=6):
                # Plot random generated images for visual evaluation of generation quality
                generated_images = self.generate(num_images=num_rows * num_cols, image_size=32,
                                                 diffusion_steps=20).cpu()
                show_image(generated_images[0])

        unet = UNet(32, 3, 32)
        unet.to(device)
        model = DiffusionModel(unet, offset_cosine_diffusion_schedule)

        return unet, model

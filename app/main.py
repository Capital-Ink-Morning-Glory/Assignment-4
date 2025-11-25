import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import load_checkpoint, clip_img, train_diffusion, train_ebm
from helper_lib.generator import generate_ebm_samples
from helper_lib.model import get_model
from pathlib import Path

# Device
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Dataset
batch_size = 64
train_loader = get_data_loader('./data', batch_size=batch_size)
test_loader = get_data_loader('./data', batch_size=batch_size, train=False)

# Models
nn_energy_model, ebm = get_model('EBM')  # Energy Based Models
unet, diffusion_model = get_model('Diffusion')  # Diffusion model

# Training
# Energy Based Models
optimizer1 = torch.optim.Adam(nn_energy_model.parameters(), lr=0.0001, betas=(0.0, 0.999))
trained_ebm = train_ebm(ebm, train_loader, test_loader, optimizer1, device=device, epochs=10)

# Diffusion model
optimizer2 = torch.optim.AdamW(diffusion_model.network.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.L1Loss()
trained_diffusion = train_diffusion(diffusion_model, train_loader, test_loader, optimizer2, loss_fn, epochs=10,
                                    device=device, checkpoint_dir='./checkpoints')

# Generation
# Energy Based Models
x = torch.rand((8, 3, 32, 32), device=device) * 2 - 1  # Uniform in [-1, 1]
new_imgs = generate_ebm_samples(nn_energy_model, x, steps=256, step_size=10.0, noise_std=0.01)

# Diffusion model
load_checkpoint(diffusion_model, optimizer2, './checkpoints/diffusion_best.pth', device=device)
diffusion_model.eval()
samples = diffusion_model.generate(num_images=1, image_size=32, diffusion_steps=1000)  # Returns tensor in [0, 1]

# Convert to numpy for plotting
image = samples[0].cpu()
image = image.detach()

# Save images
samples = clip_img(samples)
samples = samples.cpu()
fig, axes = plt.subplots(1, 8, figsize=(8 * 2, 2))
for i in range(min(8, samples.size(0))):
    img = samples[i].permute(1, 2, 0).squeeze()  # CHW to HWC
    img = img.detach().cpu().numpy()
    axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[i].axis("off")
    plt.imsave('Energy Based Models Generation.jpg', img)

plt.imsave('Diffusion Generation.jpg', image.permute(1, 2, 0).numpy())

# API
app = FastAPI()


@app.get("/image1")
async def get_image1():
    path = Path("Energy Based Models Generation.jpg")
    if not path.is_file():
        return {"error": "Image 1 not found on the server!"}
    return FileResponse(path)


@app.get("/image2")
async def get_image2():
    path = Path("Diffusion Generation.jpg")
    if not path.is_file():
        return {"error": "Image 2 not found on the server!"}
    return FileResponse(path)

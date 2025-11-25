import matplotlib.pyplot as plt
import torch


# Energy Model
def generate_ebm_samples(model, inp_imgs, steps, step_size, noise_std):
    model.eval()

    """
    As we do various calculations on the input images (like adding noise), these are added to the computational graph,
    however we really only need this for computing gradients during backpropagation.

    Energy: (x, w) => E(x, w)
    For sampling we fix the weights, and perform gradient descent with derivatives with respect to x
    for w in nn_energy_model.parameters():
        w.requires_grad = False
    inp_imgs = inp_imgs.detach().requires_grad_(True)
    """

    for _ in range(steps):
        """
        We add noise to the input images, but we will need to calculate the gradients with the transformed noisy images, 
        so tell pytorch not to track the gradient yet, this way we can avoid unnecessary computations that pytorch does 
        in order to calculate the gradients later:
        """
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        # Compute energy and gradients
        energy = model(inp_imgs)

        """
        The gradient with respect to parameters is usually done automatically when we train a neural network as part of 
        .backward() call. Here we do it manually and specify that the gradient should be with respect to the input 
        images, not the parameters. In addition because energy contains energy values for each input image in a batch, 
        we need to specify an extra grad_outputs argument for the right gradients to be calculated for each input image.
        """
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        # Finally, apply gradient clipping for stabilizing the sampling
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


def generate_diffusion_samples(model, num_img, image_size, diffusion_steps):
    # Generate images
    model.eval()
    samples = model.generate(num_images=num_img, image_size=image_size,
                             diffusion_steps=diffusion_steps)  # Returns tensor in [0, 1]

    # Convert to numpy for plotting
    image = samples[0].cpu()

    # Plot
    plt.figure(figsize=(4, 3))
    plt.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.show()

from matplotlib import pyplot as plt
import torch
import torchvision.transforms.functional as F


def show_images(data_loader, num_images=5, name="sanity_check"):
    """Display a few images from the data loader.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The data loader to get the images from.
    num_images : int
        The number of images to display.
    name : str
        The name of the file to save the plot to.
    """
    # - get the first batch of images
    iteration = iter(data_loader)
    # inputs = next(iteration)
    # - create the plot
    rows = int(num_images**0.5)
    cols = (num_images // rows) + (num_images % rows > 0)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, squeeze=True, figsize=(15, 15))
    axs = axs.flatten()

    # - show the images
    for i in range(num_images):
        inputs = next(iteration)["image"]
        i_un = inputs[i].detach().cpu()
        i_img = i_un * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)
        i_img = F.to_pil_image(i_img)
        axs[i].imshow(i_img)
        axs[i].axis("off")

    # - remove any unused subplots
    for j in range(num_images, len(axs)):
        fig.delaxes(axs[j])
    plt.close(fig)
    return fig

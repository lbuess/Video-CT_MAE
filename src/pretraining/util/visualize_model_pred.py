import matplotlib.pyplot as plt
import torch


def plot_sample(visualization: torch.tensor, out_dir: str, epoch: int) -> plt.figure:
    """
    Plot sample

    Args:
        visualization (torch.tensor): Visualization containing the image, masked image, and reconstruction
        out_dir (str): Output directory
        epoch (int): Current epoch

    Returns:
        plt.figure: Figure with the visualization
    """

    # permute tensor and move to cpu
    visualization = visualization.permute(0, 2, 3, 4, 1).detach().cpu()
    num_rows = visualization.shape[0]
    num_cols = visualization.shape[1]

    # create figure (one row for image, masked, and reconstruction --> one column per time step)
    f, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(50, 20))

    # plot images
    for i in range(num_rows):
        for j in range(num_cols):
            ax[i][j].axis("off")
            ax[i][j].imshow(
                (visualization[i][j] * 255).int(), cmap="gray", vmin=0, vmax=255
            )

    plt.savefig(out_dir + f"/eval_epoch{epoch}.png")
    return f


def visualize_model_pred(
    model: torch.nn.Module,
    mask_ratio: float,
    data_loader: torch.utils.data.DataLoader,
    out_dir: str,
    epoch: int,
    device: str,
) -> plt.figure:
    """
    Visualize model reconstruction prediction

    Args:
        model (torch.nn.Module): Model
        mask_ratio (float): Mask ratio
        video_loader (torch.utils.data.DataLoader): Data loader
        out_dir (str): Output directory
        epoch (int): Current epoch
        device (str): Device

    Returns:
        plt.figure: Figure with the visualization
    """
    fig = None
    for batch in data_loader:
        image = batch["image"].to(device)
        _, _, _, visualization = model(image, mask_ratio=mask_ratio, visualize=True)
        fig = plot_sample(visualization[0], out_dir, epoch)
        break  # only visualize one sample

    return fig

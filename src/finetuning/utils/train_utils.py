""" Utility functions for training the model """

import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from monai.data import DataLoader
from omegaconf import DictConfig
from torch.cuda.amp import autocast
from torchmetrics import AUROC, F1, Accuracy, AveragePrecision, Precision, Recall
from tqdm import tqdm


def train_one_epoch(
    train_loader: DataLoader,
    model: torch.nn.Module,
    ce_loss: torch.nn.modules.loss,
    scaler: torch.cuda.amp.grad_scaler,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    epoch: int,
    device: str,
    writer: torch.utils.tensorboard.SummaryWriter,
) -> dict:
    """
    Train the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for the training set
        model (torch.nn.Module): The model to train
        ce_loss (torch.nn.modules.loss): The loss function
        scaler (torch.cuda.amp.grad_scaler): The gradient scaler
        optimizer (torch.optim): The optimizer
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler
        device (str): The device to perform the computation on, 'cpu' or 'cuda'
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer

    Returns:
        dict: The training loss
    """
    # set model to train mode
    model.train()

    # progress bar
    training_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

    num_steps_per_epoch = len(train_loader)
    num_updates = epoch * num_steps_per_epoch
    train_loss = 0.0
    for step, batch in enumerate(training_bar):
        # log current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning rate", current_lr, num_updates)

        # get image and label from batch
        image, label = batch["image"].to(device), batch["fx"].to(device)

        # forward pass (use autocast for mixed precision)
        with autocast():
            prediction = model(image)
            loss = ce_loss(prediction, label)

        # backward pass
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # progress bar
        train_loss += loss.item()
        train_loss_average = train_loss / (step + 1)
        training_bar.set_postfix(loss=train_loss_average)

        # update learning rate scheduler
        num_updates += 1
        scheduler.step_update(num_updates=num_updates)

    # update learning rate scheduler
    scheduler.step(epoch + 1)

    return {"loss": train_loss_average}


def validate_one_epoch(
    valid_loader: DataLoader,
    model: torch.nn.Module,
    ce_loss: torch.nn.modules.loss,
    epoch: int,
    device: str,
    cfg: DictConfig,
    progress_bar_prefix: str = "Valid",
) -> dict:
    """
    Validate the model for one epoch.

    Args:
        valid_loader (DataLoader): DataLoader for the validation set
        model (torch.nn.Module): The model to validate
        ce_loss (torch.nn.modules.loss): The loss function
        epoch (int): The current epoch
        device (str): The device to perform the computation on, 'cpu' or 'cuda'
        cfg (DictConfig): Configuration file

    Returns:
        dict: The validation loss and metrics
    """
    # set model to eval mode
    model.eval()

    # progress bar
    val_bar = tqdm(valid_loader, desc=f"{progress_bar_prefix} Epoch {epoch}")

    val_loss = 0.0

    # Instantiate metrics (for our use case num_classes is fixed to 2)
    accuracy = Accuracy(num_classes=2).to(device)
    precision = Precision(num_classes=2, average="none").to(device)
    recall = Recall(num_classes=2, average="none").to(device)
    f1 = F1(num_classes=2, average="none").to(device)
    avg_precision = AveragePrecision(num_classes=2).to(device)
    auroc = AUROC(num_classes=2, average=None).to(device)

    with torch.no_grad():
        for step, batch in enumerate(val_bar):
            # get image and label from batch
            image, label = batch["image"].to(device), batch["fx"].to(device)

            # forward pass (use autocast for mixed precision)
            with autocast():
                prediction = (
                    model(image)
                    if not cfg.ensemble_strategy
                    else model.predict_ensemble(image)
                )
                loss = ce_loss(prediction, label)

            # progress bar
            val_loss += loss.item()
            val_loss_average = val_loss / (step + 1)
            val_bar.set_postfix(loss=val_loss_average)

            # Update metrics
            preds_softmax = torch.softmax(prediction, dim=1)
            accuracy.update(preds_softmax, label)
            precision.update(preds_softmax, label)
            recall.update(preds_softmax, label)
            f1.update(preds_softmax, label)
            avg_precision.update(preds_softmax, label)
            auroc.update(preds_softmax, label)

    # Compute metrics
    stats = {
        "loss": val_loss_average,
        "accuracy": accuracy.compute(),
        "precision": precision.compute()[1],
        "recall": recall.compute()[1],
        "f1": f1.compute()[1],
        "avg_precision": avg_precision.compute()[1],
        "auroc": auroc.compute()[1],
    }

    # Reset metrics for next use
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    avg_precision.reset()
    auroc.reset()

    return stats


def log_metrics(
    prefix: str,
    stats: dict,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter,
    print_metrics: bool = False,
):
    """
    Log metrics to TensorBoard and optionally print them.

    Args:
        prefix (str): The prefix for the metrics (e.g. "Train" or "Valid")
        stats (dict): The metrics to log
        epoch (int): The current epoch
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer
    """
    # Adjusted to log new metrics
    metrics_dict = {
        f"{prefix}/Loss": stats["loss"],
        f"{prefix}/Accuracy": stats["accuracy"],
        f"{prefix}/Precision": stats["precision"],
        f"{prefix}/Recall": stats["recall"],
        f"{prefix}/F1": stats["f1"],
        f"{prefix}/AvgPrecision": stats["avg_precision"],
        f"{prefix}/AUC": stats["auroc"],
    }

    if print_metrics:
        print(f"{prefix} Metrics", metrics_dict)

    # Log each metric to TensorBoard
    for metric_name, metric_value in metrics_dict.items():
        writer.add_scalar(metric_name, metric_value, epoch)


def save_model(save_path: str, model: torch.nn.Module, epoch: int, cfg: DictConfig):
    """
    Save the model to the specified path.

    Args:
        save_path (str): The path to save the model to
        model (torch.nn.Module): The model to save
        epoch (int): The current epoch
        cfg (DictConfig): Configuration file
    """
    # create save path if not exists
    save_path = Path(save_path)
    if not save_path.is_dir():
        save_path.mkdir(parents=True)

    # save model
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "cfg": cfg,
        },
        save_path / "best_model.pt",
    )


def calculate_class_weights(
    data_loader: torch.utils.data.DataLoader, device: str
) -> torch.Tensor:
    """
    Calculates the class weights inversely proportional to the frequency of each class in the dataset.

    Args:
        data_loader (DataLoader): DataLoader for the dataset.
        device (str): The device to perform the computation on, 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: A tensor of class weights.
    """

    # Initialize a tensor to count occurrences of each class.
    # Assuming binary classification; adjust the size for more classes.
    class_counts = torch.zeros(2, device=device)

    # Iterate over the DataLoader to count each class's occurrences.
    for batch in data_loader:
        labels = batch["fx"]
        labels = labels.to(device)
        for label in labels:
            class_counts[label.long()] += 1

    # Calculate class weights as inversely proportional to class frequencies.
    class_weights = 1.0 - (class_counts / class_counts.sum())

    # Log the calculated class weights for reference.
    logging.info(f"Loss class weights: {class_weights}")

    return class_weights


def test_model(
    cfg: DictConfig,
    test_loader: DataLoader,
    ce_loss: torch.nn.Module,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: str,
):
    """
    Test the model on the test set.

    Args:
        cfg (DictConfig): Configuration file
        test_loader (DataLoader): DataLoader for the test set
        device (str): The device to perform the computation on, 'cpu' or 'cuda'
    """
    # load model
    model_state_dict = torch.load(
        Path(cfg.save_dir) / "best_model.pt", map_location=device
    )
    model = instantiate(cfg.model, pretrained_weights="")
    model.to(device)
    load_status = model.load_state_dict(model_state_dict["state_dict"])
    logging.info(f"Loading weights status: {load_status}")

    # test on test set
    test_stats = validate_one_epoch(test_loader, model, ce_loss, 0, device, cfg, "Test")

    # log and print metrics
    log_metrics("Test", test_stats, 0, writer, True)

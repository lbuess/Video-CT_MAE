""" Pytorch training loop for the classification downstream task """

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from src.finetuning.data import get_data_loaders
from src.finetuning.utils.train_utils import (
    calculate_class_weights,
    log_metrics,
    save_model,
    test_model,
    train_one_epoch,
    validate_one_epoch,
)


def train(cfg: DictConfig) -> None:
    """
    Pytorch training loop for the classification downstream task.

    Args:
        cfg (dict): Configuration file
    """
    logging.info("Start finetuning ...")

    # define tensorboard writer
    writer = SummaryWriter(cfg.log_dir)

    # set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on {device} ...")

    # get training, validation and test data
    train_loader, valid_loader, test_loader = get_data_loaders(cfg)

    # get model
    model = instantiate(cfg.model)
    model.to(device)

    # define optimizer
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # define learning rate scheduler
    scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer, t_initial=cfg.epochs)

    # define loss function for training and evaluation
    class_weights = calculate_class_weights(train_loader, device)
    ce_loss = instantiate(cfg.loss, weight=class_weights if cfg.class_weights else None)
    ce_loss_eval = instantiate(cfg.loss)

    # set up gradient scaler
    scaler = GradScaler()

    # training loop
    best_val_f1 = -float("inf")
    early_stopping_counter = 0
    for epoch in range(cfg.epochs):
        # training step
        train_stats = train_one_epoch(
            train_loader,
            model,
            ce_loss,
            scaler,
            optimizer,
            scheduler,
            epoch,
            device,
            writer,
        )

        # validation step
        valid_stats = validate_one_epoch(
            valid_loader, model, ce_loss_eval, epoch, device, cfg
        )

        # log metrics to tensorboard
        writer.add_scalar("Train/Loss", train_stats["loss"], epoch)
        log_metrics("Valid", valid_stats, epoch, writer)

        # check for early stopping
        if valid_stats["f1"] > best_val_f1:
            best_val_f1 = valid_stats["f1"]
            early_stopping_counter = 0
            logging.info(f"Epoch {epoch}: Saving model with F1 score {best_val_f1} ...")
            save_model(cfg.save_dir, model, epoch, cfg)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= cfg.early_stopping_patience:
            logging.info("Early stopping triggered. Stopping training.")
            break

    # test model on test set
    if cfg.test_after_training:
        logging.info("Testing model on test set ...")
        test_model(cfg, test_loader, ce_loss_eval, writer, device)

    # training finished
    logging.info("Finished training.")

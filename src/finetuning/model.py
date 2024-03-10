""" Model definition for the Vision Transformer (ViT) model """

import logging

import torch
import torch.nn as nn

from src.finetuning.utils.model_utils import vit_large_patch16


class ViT(nn.Module):
    """
    Vision Transformer model.
    """

    def __init__(
        self,
        input_size=96,
        num_frames=16,
        t_patch_size=2,
        num_classes=2,
        cls_embed=True,
        sep_pos_embed=True,
        pos_enc_init="pretrained_pos",
        pretrained_weights="",
    ):
        super().__init__()
        logging.info("Initializing model ...")

        self.model = vit_large_patch16(
            img_size=input_size,
            num_frames=num_frames,
            t_patch_size=t_patch_size,
            num_classes=num_classes,
            cls_embed=cls_embed,
            sep_pos_embed=sep_pos_embed,
        )
        self.sep_pos_embed = sep_pos_embed
        self.pos_enc_init = pos_enc_init
        self.input_size = input_size
        self.patch_size = 16  # for this implementation fixed to 16

        # load pretrained weights if specified
        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Predictions of the model
        """
        out = self.model(x)
        return out

    def predict_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the ensemble of the input tensor.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Predictions of the ensemble
        """
        x = x.squeeze(dim=0)  # remove batch dimension
        out_ensemble = self.model(x)  # forward pass for all elements in the ensemble
        out_ensemble = out_ensemble.mean(dim=0, keepdim=True)  # average the predictions
        return out_ensemble

    def load_pretrained_weights(self, weight_path: str) -> None:
        """
        Loads pretrained weights into the model.

        Args:
            weight_path (str): Path to the file containing the pretrained weights
        """
        logging.info(
            f"Loading pretrained weights from {weight_path} using {self.pos_enc_init} initialization ..."
        )

        # load the pretrained weights
        state_dict = torch.load(weight_path, map_location="cpu")

        # check if the model_state key exists and adjust the state_dict accordingly
        state_dict = state_dict.get("model_state", state_dict.get("model"))

        # adjust the state_dict based on the positional encoding initialization method
        if self.pos_enc_init == "pretrained":
            adjusted_state_dict = state_dict

        elif self.pos_enc_init == "crop":
            adjusted_state_dict = self.crop_pretrained_pos_enc(state_dict)

        elif self.pos_enc_init == "random":
            adjusted_state_dict = self.skip_pos_enc(state_dict)

        # load the adjusted state_dict into the model
        load_status = self.model.load_state_dict(adjusted_state_dict, strict=False)
        logging.info(f"Loading weights status: {load_status}")

    def crop_pretrained_pos_enc(self, state_dict: dict) -> dict:
        """
        Adjusts the pretrained positional encoding tensor to fit the current model's dimensions.

        Args:
            state_dict (dict): The loaded state dictionary to adjust

        Returns:
            dict: The adjusted state dictionary with the updated positional encoding
        """
        # get number of patches along each dimension for the original and new sizes
        orig_patches_per_dim = 224 // self.patch_size  # original 224x224 model
        new_patches_per_dim = self.input_size // self.patch_size

        # reshape positional encodings to fit patch dimensions (224x224 frame size --> 14x14 patches)
        pos_enc = state_dict["pos_embed_spatial"].reshape(
            1, orig_patches_per_dim, orig_patches_per_dim, -1
        )

        # get cropping indices dynamically based on the new and original dimensions
        crop_start = (orig_patches_per_dim - new_patches_per_dim) // 2
        crop_end = crop_start + new_patches_per_dim

        # crop positional encodings to fit the current model's dimensions
        cropped_pos_enc = pos_enc[
            :, crop_start:crop_end, crop_start:crop_end, :
        ].reshape(1, new_patches_per_dim**2, -1)

        # update the state_dict with the cropped positional encodings
        state_dict["pos_embed_spatial"] = cropped_pos_enc

        return state_dict

    def skip_pos_enc(self, state_dict: dict) -> dict:
        """
        Initializes positional encoding randomly based on the current model's state_dict and the loaded state_dict shapes.

        Args:
            state_dict (dict): The loaded state dictionary to adjust.

        Returns:
            dict: The model's adjusted state dictionary with randomized positional encoding where applicable.
        """
        # state dict of the model we are loading into
        model_state_dict = self.model.state_dict()

        # skip positional encoding layers if shape mismatch
        for name, param in state_dict.items():
            if name in model_state_dict and model_state_dict[name].shape != param.shape:
                logging.info(f"Skipping {name} due to shape mismatch ...")
                continue
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
        return model_state_dict

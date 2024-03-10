import logging
import sys

import hydra
from omegaconf import DictConfig

from src.finetuning.train import train as start_finetuning
from src.pretraining.main_pretrain import main as start_pretraining

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@hydra.main(version_base=None, config_path="config/finetuning", config_name="config")
def run_finetuning(cfg: DictConfig) -> None:
    """
    Run the finetuning task specified in the config file.

    Args:
        cfg (DictConfig): Configuration file
    """
    start_finetuning(cfg)


@hydra.main(version_base=None, config_path="config/pretraining", config_name="config")
def run_pretraining(cfg: DictConfig) -> None:
    """
    Run the pretraining task specified in the config file.

    Args:
        cfg (DictConfig): Configuration file

    Raises:
        NotImplementedError: If the function is called
    """
    start_pretraining(cfg)


if __name__ == "__main__":
    # Run either finetuning or pretraining
    if "task=finetuning" in sys.argv:
        run_finetuning()
    else:
        run_pretraining()

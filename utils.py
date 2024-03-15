import argparse
import random
from pathlib import Path

import numpy as np
import torch


def setup_argument_parser(working_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the arrow dataset for training"
    )

    parser.add_argument(
        "--n", type=int, default=10, help="Number of background images to to use"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of arrows to place on each background image",
    )

    parser.add_argument(
        "--arrow-dir",
        type=str,
        default=str(working_dir / "arrows"),
        help="Directory containing arrow images",
    )

    parser.add_argument(
        "--background-dir",
        type=str,
        default=str(working_dir / "backgrounds"),
        help="Directory to download the background images to",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(working_dir / "dataset"),
        help="Directory to save the generated dataset",
    )

    return parser


def setup_directories(args: argparse.Namespace) -> None:
    if Path(args.arrow_dir).exists() is False:
        raise FileNotFoundError(f"{args.arrow_dir} does not exist")

    Path(args.background_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def set_randomness(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

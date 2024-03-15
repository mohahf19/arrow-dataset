from pathlib import Path

import datasets
import numpy as np
import requests
from datasets.utils.file_utils import get_datasets_user_agent
from loguru import logger

USER_AGENT = get_datasets_user_agent()


def download_image(url: str, retries: int) -> bytes | None:
    for _ in range(retries):
        try:
            img_data = requests.get(url).content
            return img_data
        except Exception:
            logger.error(f"Failed to download {url}")
            continue
    return None


def download_background_images(background_dir: str, n: int) -> None:
    logger.info(f"Downloading {n} images to {background_dir}...")

    # Download images here
    dataset = datasets.load_dataset("red_caps", "earthporn")["train"]
    dataset_length = len(dataset)
    random_inds = np.random.permutation(dataset_length)

    counter = 0
    curr_ind = 0

    while counter < n and curr_ind < dataset_length:
        ind = int(random_inds[curr_ind])
        image_url = dataset[ind]["image_url"]
        image_id = dataset[ind]["image_id"]
        image_file_path = f"{background_dir}/{image_id}.jpg"

        if not Path(image_file_path).exists():
            img_bytes = download_image(image_url, retries=3)
            # 2048 is a heuristic to check if the image is valid
            if img_bytes is not None and len(img_bytes) > 2048:  # noqa: PLR2004
                with open(image_file_path, "wb") as f:
                    f.write(img_bytes)
                counter += 1
        else:
            counter += 1

        logger.info(f"Downloaded {counter} / {n} images")
        curr_ind += 1

    logger.info("Done!")

import random
from enum import Enum
from pathlib import Path

import numpy as np
import torchvision
from loguru import logger
from torch import Tensor
from tqdm import tqdm

NUM_DIRECTIONS = 8

ANGLES = np.linspace(0, 360, NUM_DIRECTIONS, endpoint=False)


class Direction(Enum):
    NORTH = 0
    NORTH_WEST = 45
    WEST = 90
    SOUTH_WEST = 135
    SOUTH = 180
    SOUTH_EAST = 225
    EAST = 270
    NORTH_EAST = 315

    def __str__(self):
        return str(self.name).lower()


class Arrow:
    image: Tensor
    name: str
    path: Path

    def __init__(self, image: Tensor, name: str, path: Path):
        self.image = image
        self.name = name
        self.path = path


class LabeledImage:
    image: Tensor
    label: Direction
    unique_id: str

    def __init__(self, image: Tensor, label: Direction, unique_id: str):
        self.image = image
        self.label = label
        self.unique_id = unique_id


class DatasetAssembler:
    def __init__(  # noqa: PLR0913
        self,
        background_dir: str,
        arrow_dir: str,
        output_dir: str,
        k: int,
        image_size: tuple[int, int] = (128, 128),
        arrow_size: tuple[int, int] = (32, 32),
    ):
        self.background_dir = Path(background_dir)
        self.arrow_dir = Path(arrow_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.arrow_size = arrow_size

        self._image_resize = torchvision.transforms.Resize(self.image_size)
        self._arrow_resize = torchvision.transforms.Resize(self.arrow_size)
        self._rotate = torchvision.transforms.functional.rotate
        self._read_img = lambda x: torchvision.io.read_image(
            str(x), mode=torchvision.io.image.ImageReadMode.RGB
        )

        for directory in [self.background_dir, self.arrow_dir, self.output_dir]:
            if directory.exists() is False:
                raise FileNotFoundError(f"{dir} does not exist")

        self.k = k

    def load_arrows(self) -> list[Arrow]:
        arrows = [
            Arrow(
                self._arrow_resize(self._read_img(arrow_path)),
                arrow_path.stem,
                arrow_path,
            )
            for arrow_path in self.arrow_dir.glob("*.png")
        ]
        return arrows

    def load_background_img(self, image_path: Path) -> Tensor:
        return self._image_resize(self._read_img(image_path))

    def combine_background_with_arrow(
        self, arrow: Arrow, background: Tensor, unique_id: str
    ) -> list[LabeledImage]:
        assert arrow.image.shape[1:] == self.arrow_size
        assert background.shape[1:] == self.image_size

        labeled_images = []
        for ind in range(self.k):
            # Pick a random rotation for the arrow
            angle = random.choice(ANGLES)
            rotated_arrow = self._rotate(arrow.image.clone(), angle)

            # Place the arrow on the background
            x = random.randint(0, self.image_size[0] - self.arrow_size[0])
            y = random.randint(0, self.image_size[1] - self.arrow_size[1])

            background_with_arrow = background.clone()

            # I want to overlay transparent arrow on top of the background
            # Only the arrow part should be visible, using the alpha channel
            arrow_mask = rotated_arrow > 0

            background_with_arrow[
                :, y : y + self.arrow_size[1], x : x + self.arrow_size[0]
            ][arrow_mask] = rotated_arrow[arrow_mask]

            direction = Direction(angle)
            labeled_images.append(
                LabeledImage(
                    background_with_arrow,
                    direction,
                    f"{unique_id}_arrow_{arrow.name}_iter_{ind}_{str(direction)}",
                )
            )

        return labeled_images

    def combine_background_with_arrows(
        self, arrows: list[Arrow], background: Tensor, image_name: str
    ) -> list[LabeledImage]:
        return [
            labeled_image
            for arrow_ind, arrow in enumerate(arrows)
            for labeled_image in self.combine_background_with_arrow(
                arrow, background, image_name
            )
        ]

    def save_labels(self, labels: dict[str, Direction]) -> None:
        with open(self.output_dir / "labels.csv", "w") as f:
            f.write("unique_id,direction\n")
            for unique_id, direction in labels.items():
                f.write(f"{unique_id},{direction}\n")

    def assemble_and_save_dataset(self) -> None:
        arrows = self.load_arrows()
        labels: dict[str, Direction] = {}
        logger.info("Processing..")
        paths = list(self.background_dir.glob("*.jpg"))
        for background_path in tqdm(paths, desc="Backgrounds"):
            background = self.load_background_img(background_path)

            labeled_images = self.combine_background_with_arrows(
                arrows, background, background_path.stem
            )

            for labeled_image in labeled_images:
                str(labeled_image.label)
                torchvision.io.write_png(
                    labeled_image.image,
                    str(self.output_dir / f"{labeled_image.unique_id}.png"),
                )
                if labeled_image.unique_id in labels:
                    raise ValueError(
                        f"Duplicate unique_id found: {labeled_image.unique_id}"
                    )
                labels[labeled_image.unique_id] = labeled_image.label
        self.save_labels(labels)

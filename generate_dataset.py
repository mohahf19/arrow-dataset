from pathlib import Path

from loguru import logger

from dataset_assembler import DatasetAssembler
from download_utils import download_background_images
from utils import set_randomness, setup_argument_parser, setup_directories

if __name__ == "__main__":
    set_randomness(42)

    working_dir = Path(__file__).parent
    parser = setup_argument_parser(working_dir)
    args = parser.parse_args()

    logger.info(f"Arguments: {vars(args)}")
    setup_directories(args)

    download_background_images(args.background_dir, args.n)

    dataset_assembler = DatasetAssembler(
        args.background_dir, args.arrow_dir, args.output_dir, args.k
    )

    dataset_assembler.assemble_and_save_dataset()

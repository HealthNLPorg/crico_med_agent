import argparse
import logging

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_folder", type=str)
parser.add_argument("--output_dir", type=str)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def reshape(
    input_folder: str,
    output_folder: str,
) -> None:
    pass


def main() -> None:
    args = parser.parse_args()
    reshape(args.input_folder, args.output_dir)


if __name__ == "__main__":
    main()

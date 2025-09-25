import argparse
import shutil
import os
from ..utils import mkdir
import logging

parser = argparse.ArgumentParser(description="")

parser.add_argument("--raw_iaa_dir", type=str)
parser.add_argument("--output_dir", type=str)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def reorganize(raw_iaa_dir: str, output_dir: str) -> None:
    matches = [
        ("dave",),
        ("yang",),
        ("dave", "adjudication"),
        ("yang", "adjudication"),
    ]

    def right_annotator(dest: tuple[str, ...], lower_fn: str) -> bool:
        if not fn.endswith("xml"):
            return True  # leave the source files by default
        match dest:
            case (name,):
                return name in lower_fn and "adjudication" not in lower_fn
            case (name, adj):
                return name in lower_fn and adj in lower_fn
            case _:
                ValueError(f"Issue with {dest}")
                return False

    for dest in matches:
        dest_path = os.path.join(output_dir, "_".join(dest))
        mkdir(dest_path)
        for dirname in os.listdir(raw_iaa_dir):
            src_path = os.path.join(raw_iaa_dir, dirname)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, os.path.join(dest_path, dirname))
        for root, dirs, files in os.walk(dest_path):
            for fn in files:
                lower_fn = fn.strip().lower()
                if not right_annotator(dest, lower_fn):
                    os.remove(os.path.join(root, fn))


def main() -> None:
    args = parser.parse_args()
    reorganize(args.raw_iaa_dir, args.output_dir)


if __name__ == "__main__":
    main()

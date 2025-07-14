import argparse
import logging
import os
import shutil
from functools import partial

from ..utils import basename_no_ext, mkdir

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_folder", type=str)
parser.add_argument("--output_dir", type=str)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def make_new_directory_and_copy_file(
    input_folder: str, output_folder: str, fn: str
) -> None:
    anafora_identifier = basename_no_ext(fn).split(".")[0]
    identifier_dir = os.path.join(output_folder, anafora_identifier)
    mkdir(identifier_dir)
    source_file_path = os.path.join(input_folder, fn)
    dest_file_path = os.path.join(identifier_dir, fn)
    shutil.copyfile(source_file_path, dest_file_path)


def reshape(
    input_folder: str,
    output_folder: str,
) -> None:
    def is_xml(fn: str) -> bool:
        return fn.lower().strip().endswith("xml")

    make_new_dir_then_copy = partial(
        make_new_directory_and_copy_file, input_folder, output_folder
    )
    for xml_fn in filter(is_xml, os.listdir(input_folder)):
        make_new_dir_then_copy(xml_fn)


def main() -> None:
    args = parser.parse_args()
    reshape(args.input_folder, args.output_dir)


if __name__ == "__main__":
    main()

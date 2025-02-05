import argparse
import logging
import os
import re
from ast import literal_eval
from typing import Iterable

import pandas as pd

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--tsv_dir",
    type=str,
    help="Dir full of TSVs to verify",
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# retain newline information via special markers
# while removing them for storage
# ( so you can load them later via pandas without parsing errors )
# ignoring vertical tabs (\v) for now
# unless we run into them
def serialize_whitespace(sample: str) -> str:
    return (
        sample.replace("\n", "<cn>")
        .replace("\t", "<ct>")
        .replace("\f", "<cf>")
        .replace("\r", "<cr>")
    )


def deserialize_whitespace(sample: str) -> str:
    return (
        sample.replace("<cn>", "\n")
        .replace("<ct>", "\t")
        .replace("<cf>", "\f")
        .replace("<cr>", "\r")
    )


def is_valid(frame: pd.DataFrame) -> bool:
    frame["medication offsets"] = frame["medication offsets"].apply(literal_eval)
    frame["window offsets"] = frame["window offsets"].apply(literal_eval)

    def deserialized_instance(row: pd.Series, annotation_type: str) -> str:
        return re.sub(
            "</?medication>",
            "",
            deserialize_whitespace(row[f"{annotation_type} window"]),
        )

    def window_length_is_valid(row: pd.Series) -> bool:
        cas_level_begin, cas_level_end = row["window offsets"]
        return cas_level_end - cas_level_begin == len(
            deserialized_instance(row, "window")
        )

    window_length_validity = frame.apply(window_length_is_valid, axis=1)
    if not all(window_length_validity):
        logger.info("Issues with window length validity")
        logger.info(frame.loc[window_length_validity])
        return False

    def medication_length_is_valid(row: pd.Series) -> bool:
        cas_level_begin, cas_level_end = row["medication offsets"]
        return cas_level_end - cas_level_begin == len(
            deserialized_instance(row, "medication")
        )

    medication_length_validity = frame.apply(medication_length_is_valid, axis=1)
    if not all(medication_length_validity):
        logger.info("Issues with medication length validity")
        logger.info(frame.loc[medication_length_validity])
        return False

    def deserialized_medication_indices_are_valid(row: pd.Series) -> bool:
        cas_level_begin, cas_level_end = row["medication offsets"]
        return False


def get_frames(tsv_dir: str) -> Iterable[pd.DataFrame]:
    for fn in os.listdir(tsv_dir):
        if fn.lower().endswith("tsv"):
            df = pd.read_csv(os.path.join(tsv_dir, fn), sep="\t")
            yield df


def main() -> None:
    args = parser.parse_args()
    all_frames_valid = True
    for frame in get_frames(args.tsv_dir):
        if not is_valid(frame):
            all_frames_valid = False
    (
        logger.info("Success!")
        if all_frames_valid
        else logger.info("Validity issues with one or more frames")
    )


if __name__ == "__main__":
    main()

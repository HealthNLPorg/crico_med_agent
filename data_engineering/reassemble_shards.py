import argparse
import gc
import logging
import os
from collections import deque, defaultdict
from itertools import chain
from typing import Deque, Iterable

import pandas as pd

from partition_time_by_job_count_shards import mkdir

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_folders", nargs="+")
parser.add_argument("--output_dir", type=str)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_relevant_paths(input_folders: list[str]) -> tuple[Iterable[str], Iterable[str]]:
    study_id_fn = "shard_study_ids.txt"
    processed_shard_frame_fn = "shard_frame.tsv"
    processed_keyword = "processed"
    frame_paths: Deque[str] = deque()
    study_id_paths: Deque[str] = deque()
    for input_folder in input_folders:
        for root, dirs, files in os.walk(input_folder):
            if study_id_fn in files:
                study_id_paths.append(os.path.join(root, study_id_fn))
            elif processed_keyword in root:
                frame_paths.append(os.path.join(root, processed_shard_frame_fn))
    return frame_paths, study_id_paths


def build_and_write_frame(frame_paths: Iterable[str], output_dir: str) -> None:
    gc.enable()

    def get_id_number(fn: str) -> int:
        return int(fn.split("_")[-1])

    def load_frame(frame_fn: str) -> pd.DataFrame:
        return pd.read_csv(frame_fn, sep="\t")

    full_frame = pd.concat(map(load_frame, frame_paths))
    gc.collect()
    full_frame["int_study_id"] = full_frame["filename"].map(get_id_number)
    full_frame = full_frame.sort_values(by="int_study_id")
    full_frame.drop(columns=["int_study_id"], inplace=True)
    assert not full_frame.duplicated().any()
    full_frame.to_csv(
        os.path.join(output_dir, "merged_shards.tsv"), sep="\t", index=False
    )


def build_and_write_study_ids(study_id_paths: Iterable[str], output_dir: str) -> None:
    id_to_files = defaultdict(deque)

    def load_ids(study_id_path: str) -> Iterable[int]:
        with open(study_id_path, mode="r", encoding="utf-8") as f:
            return map(int, f.readlines())

    all_inds = sorted(chain.from_iterable(map(load_ids, study_id_paths)))
    assert len(all_inds) == len(set(all_inds))
    with open(
        os.path.join(output_dir, "merged_shards_study_ids.txt"),
        mode="w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(
                map(str, sorted(chain.from_iterable(map(load_ids, study_id_paths))))
            )
        )


def process(input_folders: list[str], output_dir: str) -> None:
    frame_paths, study_id_paths = get_relevant_paths(input_folders)
    mkdir(output_dir)
    build_and_write_frame(frame_paths, output_dir)
    build_and_write_study_ids(study_id_paths, output_dir)


def main() -> None:
    args = parser.parse_args()
    process(args.input_folders, args.output_dir)


if __name__ == "__main__":
    main()

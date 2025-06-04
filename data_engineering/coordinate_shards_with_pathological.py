import argparse
import gc
import os
from typing import Iterable

import pandas as pd


parser = argparse.ArgumentParser(description="")
parser.add_argument("--shard_folder_with_fixed", type=str)
parser.add_argument("--remainder_shards_bundled", type=str)
parser.add_argument("--empty_meds_preimage_tsv", type=str)
parser.add_argument("--output_dir", type=str)


def __section_hash(
    base_filename: str,
    section_identifier: str,
    # section_begin: int, section_end: int
) -> str:
    return f"{base_filename.strip()} - {section_identifier.strip()}"
    # - {section_begin},{section_end}"


def __tsv_to_section_hashes(target_tsv: str) -> set[str]:
    # assumes format of the 'preimage' frames
    # def __local_section_hash(row: pd.Series) -> str:
    #     section_begin, section_end = [
    #         int(i.strip()) for i in row.section_offsets.split(",")
    #     ]
    #     return __section_hash(
    #         base_filename=row.filename,
    #         section_identifier=row.section_identifier,
    #         section_begin=section_begin,
    #         section_end=section_end,
    #     )

    df = pd.read_csv(target_tsv, sep="\t")
    return set(
        df.apply(
            __section_hash,
            # __local_section_hash,
            axis=1,
        ).to_list()
    )


def get_relevant_paths(remainder_folders: str) -> Iterable[str]:
    for input_folder in os.listdir(remainder_folders):
        for root, dirs, files in os.walk(input_folder):
            if "processed" in root.lower():
                for fn in files:
                    if fn.lower().endswith("tsv"):
                        yield os.path.join(remainder_folders, input_folder, fn)


def __build_remainder_frame(
    remainder_shards_bundled: str,
    empty_meds_preimage_tsv: str,
) -> pd.DataFrame:
    problem_sections = __tsv_to_section_hashes(empty_meds_preimage_tsv)

    def __is_not_problem_section(row: pd.Series) -> bool:
        return (
            __section_hash(
                base_filename=row.filename, section_identifier=row.section_identifier
            )
            not in problem_sections
        )

    gc.enable()

    def get_id_number(fn: str) -> int:
        return int(fn.split("_")[-1])

    def load_frame(frame_fn: str) -> pd.DataFrame:
        return pd.read_csv(frame_fn, sep="\t")

    full_frame = pd.concat(
        map(load_frame, get_relevant_paths(remainder_shards_bundled))
    )
    gc.collect()
    full_frame["int_study_id"] = full_frame["filename"].map(get_id_number)
    full_frame = full_frame.sort_values(by="int_study_id")
    pass


def __process(
    shard_folder_with_fixed: str,
    remainder_shards_bundled: str,
    empty_meds_preimage_tsv: str,
    output_dir: str,
) -> None:
    pass


def main() -> None:
    args = parser.parse_args()
    __process(
        args.shard_folder_with_fixed,
        args.remainder_shards_bundled,
        args.empty_meds_preimage_tsv,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

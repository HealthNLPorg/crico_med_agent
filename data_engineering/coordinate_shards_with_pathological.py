import argparse
import gc
import os
from typing import Iterable
from itertools import chain

import pandas as pd


parser = argparse.ArgumentParser(description="")
parser.add_argument("--shard_folder_with_fixed", type=str)
parser.add_argument("--remainder_shards_bundled", type=str)
parser.add_argument("--empty_meds_preimage_tsv", type=str)
parser.add_argument("--output_dir", type=str)


def __section_hash(
    base_filename: str,
    section_identifier: str,
) -> str:
    return f"{base_filename.strip()} - {section_identifier.strip()}"


def __get_relevant_paths(input_folder: str) -> Iterable[str]:
    for root, dirs, files in input_folder:
        if "processed" in root.lower():
            for fn in files:
                if fn.lower().endswith("tsv"):
                    yield os.path.join(input_folder, fn)


def __build_frame(
    shard_bundle_dirs: list[str],
) -> pd.DataFrame:
    def load_frame(frame_fn: str) -> pd.DataFrame:
        return pd.read_csv(frame_fn, sep="\t")

    full_frame = pd.concat(
        chain.from_iterable(
            map(load_frame, __get_relevant_paths(input_folder=shard_dir))
            for shard_dir in shard_bundle_dirs
            if "shard" in shard_dir
        )
    )
    gc.collect()
    return full_frame


def __process(
    shard_folder_with_fixed: str,
    remainder_shards_bundled: str,
    empty_meds_preimage_tsv: str,
    output_dir: str,
) -> None:
    gc.enable()

    raw_remainder_frame = __build_frame(
        [
            os.path.join(remainder_shards_bundled, dirname)
            for dirname in os.listdir(remainder_shards_bundled)
            if "agent_2" in dirname
        ]
    )

    def __local_section_hash(row: pd.Series) -> str:
        return __section_hash(
            base_filename=row.filename, section_identifier=row.section_identifier
        )

    problem_sections = set(
        pd.read_csv(empty_meds_preimage_tsv, sep="\t")
        .apply(__local_section_hash, axis=1)
        .to_list()
    )
    fixed_shard_frame = __build_frame(
        [
            shard_folder_with_fixed,
        ]
    )
    sections_with_fixed = set(
        fixed_shard_frame.apply(__local_section_hash, axis=1).to_list()
    )

    assert problem_sections.issubset(sections_with_fixed)

    def get_id_number(fn: str) -> int:
        return int(fn.split("_")[-1])

    def __is_not_problem_section(row: pd.Series) -> bool:
        return (
            __section_hash(
                base_filename=row.filename, section_identifier=row.section_identifier
            )
            not in problem_sections
        )

    filtered_remainder_frame = raw_remainder_frame.loc[
        raw_remainder_frame.apply(__is_not_problem_section, axis=1)
    ]
    full_frame = pd.concat((filtered_remainder_frame, fixed_shard_frame))
    gc.collect()
    full_frame["int_study_id"] = full_frame["filename"].map(get_id_number)
    full_frame = full_frame.sort_values(by="int_study_id")
    full_frame.drop(columns=["int_study_id"], inplace=True)
    full_frame.to_csv(
        os.path.join(output_dir, "all_shards_merged.tsv"), sep="\t", index=False
    )


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

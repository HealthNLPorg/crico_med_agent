import argparse
import gc
import os
from typing import Iterable
from itertools import chain

import pandas as pd


parser = argparse.ArgumentParser(description="")
parser.add_argument("--shard_folder_with_fixed", type=str)
parser.add_argument("--remainder_shards_bundled", type=str, default=None)
parser.add_argument("--empty_meds_tsv", type=str, default=None)
parser.add_argument("--output_dir", type=str)

def __get_id_number(fn: str) -> int:
    return int(fn.split("_")[-1])

def __section_hash(
    base_filename: str,
    section_identifier: str,
) -> str:
    return f"{base_filename.strip().lower()} - {section_identifier.strip().lower()}"


def __get_relevant_paths(input_folder: str) -> Iterable[str]:
    for root, dirs, files in os.walk(input_folder):
        if "processed" in root.lower():
            for fn in files:
                if fn.lower().endswith("tsv"):
                    yield os.path.join(input_folder, fn)


def __build_frame(
    shard_bundle_dirs: list[str],
) -> pd.DataFrame:
    if len(shard_bundle_dirs) == 0:
        return pd.DataFrame([])

    def load_frame(frame_fn: str) -> pd.DataFrame:
        return pd.read_csv(frame_fn, sep="\t")

    full_frame = pd.concat(
        chain.from_iterable(
            map(
                load_frame,
                __get_relevant_paths(
                    input_folder=os.path.join(shard_bundle_dir, shard_dir)
                ),
            )
            for shard_bundle_dir in shard_bundle_dirs
            for shard_dir in os.listdir(shard_bundle_dir)
            if "shard" in shard_dir
        )
    )
    gc.collect()
    return full_frame


def __process(
    shard_folder_with_fixed: str,
    remainder_shards_bundled: str | None,
    empty_meds_tsv: str | None,
    output_dir: str,
) -> None:
    gc.enable()

    original_shards_with_pathological = __build_frame(
        [
            os.path.join(remainder_shards_bundled, dirname)
            for dirname in (
                os.listdir(remainder_shards_bundled)
                if remainder_shards_bundled is not None
                else []
            )
            if "agent_2" in dirname
        ]
    )

    def __local_section_hash(row: pd.Series) -> str:
        return __section_hash(
            base_filename=row.filename, section_identifier=row.section_identifier
        )

    fixed_shard_frame = __build_frame(
        [
            shard_folder_with_fixed,
        ]
    )
    if empty_meds_tsv is not None:
        problem_sections = set(
            pd.read_csv(empty_meds_tsv, sep="\t")
            .apply(__local_section_hash, axis=1)
            .to_list()
        )
        fixed_frames_sections = set(
            fixed_shard_frame.apply(__local_section_hash, axis=1).to_list()
        )

        # original_frames_sections = (
        #     set(
        #         original_shards_with_pathological.apply(
        #             __local_section_hash, axis=1
        #         ).to_list()
        #     )
        #     if len(original_shards_with_pathological) > 0
        #     else set()
        # )

        # assert problem_sections.issubset(
        #     fixed_frames_sections
        # ), problem_sections.intersection(fixed_frames_sections)
        # assert problem_sections.issubset(original_frames_sections)


        def __is_not_problem_section(row: pd.Series) -> bool:
            return __local_section_hash(row) not in problem_sections

        # filtered_remainder_frame = original_shards_with_pathological.loc[
        #     original_shards_with_pathological.apply(__is_not_problem_section, axis=1)
        # ]
        # full_frame = pd.concat((filtered_remainder_frame, fixed_shard_frame))
        fixed_shard_frame = fixed_shard_frame.loc[
            fixed_shard_frame.apply(__is_not_problem_section, axis=1)
        ]
    full_frame = pd.concat((original_shards_with_pathological, fixed_shard_frame))
    gc.collect()
    full_frame["int_study_id"] = full_frame["filename"].map(__get_id_number)
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
        args.empty_meds_tsv,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

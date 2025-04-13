import argparse
import os
import pandas as pd
from operator import itemgetter
import gc
import logging
from collections import deque
from typing import Deque, cast
import pathlib

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_tsv", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--initial", type=int)
parser.add_argument("--job_count", type=int)
parser.add_argument("--hours_per_job", type=int, default=-1)
parser.add_argument("--minutes_per_job", type=int, default=-1)
parser.add_argument("--seconds_per_instance", type=int)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def mkdir(dir_name: str) -> None:
    _dir_name = pathlib.Path(dir_name)
    _dir_name.mkdir(parents=True, exist_ok=True)


def get_instances_per_job(
    hours_per_job: int, minutes_per_job: int, seconds_per_instance: int
) -> int:
    if (hours_per_job < 0 and minutes_per_job < 0) or (
        hours_per_job > 0 and minutes_per_job > 0
    ):
        logger.error(
            "Need to provide exactly one of a positive integer hours or minutes per job"
        )
        exit(1)
    seconds_per_job = (
        hours_per_job * 60 * 60 if hours_per_job > 0 else minutes_per_job * 60
    )
    return seconds_per_job // seconds_per_instance


def process(
    input_tsv: str,
    output_dir: str,
    initial: int,
    job_count: int,
    hours_per_job: int,
    minutes_per_job: int,
    seconds_per_instance: int,
) -> None:
    gc.enable()

    def get_id_number(fn: str) -> int:
        return int(fn.split("_")[-1])

    current_instances = 0
    shards_remaining = job_count
    shard_data: Deque[tuple[int, pd.DataFrame]] = deque()
    instances_per_job = get_instances_per_job(
        hours_per_job, minutes_per_job, seconds_per_instance
    )
    full_frame = pd.read_csv(input_tsv, sep="\t", low_memory=False)
    full_frame["int_study_id"] = full_frame["filename"].map(get_id_number)
    full_frame = full_frame.sort_values(by="int_study_id")
    # full_frame.drop(columns=["int_study_id"], inplace=True)
    for fn, fn_sub_frame in full_frame.groupby(by="int_study_id"):
        if shards_remaining == 0:
            break
        if current_instances < instances_per_job:
            reached = current_instances + len(fn_sub_frame) == instances_per_job
            if current_instances + len(fn_sub_frame) > instances_per_job or reached:
                if reached:
                    shard_data.append((cast(str, fn), fn_sub_frame))
                shard_dir = os.path.join(
                    output_dir, f"shard_{(job_count - shards_remaining) + initial}"
                )
                mkdir(shard_dir)
                df = pd.concat(map(itemgetter(1), shard_data))
                df.drop(columns=["int_study_id"], inplace=True)
                df.to_csv(
                    os.path.join(shard_dir, "shard_frame.tsv"), sep="\t", index=False
                )
                with open(
                    os.path.join(shard_dir, "shard_study_ids.txt"),
                    mode="w",
                    encoding="utf-8",
                ) as f:
                    f.write("\n".join(map(str, map(itemgetter(0), shard_data))))
                shard_data.clear()
                gc.collect()
                full_frame.drop(df.index, inplace=True)
                current_instances = 0
                shards_remaining -= 1
            else:
                shard_data.append((cast(str, fn), fn_sub_frame))
                current_instances += len(fn_sub_frame)
        else:
            logger.error("This shouldn't happen!")
            exit(1)

    full_frame.drop(columns=["int_study_id"], inplace=True)
    full_frame.to_csv(os.path.join(output_dir, "remainder.tsv"), sep="\t", index=False)


def main() -> None:
    args = parser.parse_args()
    process(
        args.input_tsv,
        args.output_dir,
        args.initial,
        args.job_count,
        args.hours_per_job,
        args.minutes_per_job,
        args.seconds_per_instance,
    )


if __name__ == "__main__":
    main()

import argparse
import logging
import os
import pathlib
import sys
from collections import deque
from functools import lru_cache, partial
from math import ceil
from operator import itemgetter
from typing import Deque, cast

import pandas as pd

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
        sys.exit(1)
    seconds_per_job = (
        hours_per_job * 60 * 60 if hours_per_job > 0 else minutes_per_job * 60
    )
    return seconds_per_job // seconds_per_instance


@lru_cache
def get_hours_for_instances(total_instances: int, seconds_per_instance) -> int:
    return ceil((total_instances * seconds_per_instance) / 360)


@lru_cache
def get_buffered_hours_for_instances(total_instances: int, seconds_per_instance) -> int:
    numerator = 5
    denominator = 4
    return ceil(
        (numerator * get_hours_for_instances(total_instances, seconds_per_instance))
        / denominator
    )


def process(
    input_tsv: str,
    output_dir: str,
    initial: int,
    job_count: int,
    hours_per_job: int,
    minutes_per_job: int,
    seconds_per_instance: int,
) -> None:
    def get_id_number(fn: str) -> int:
        return int(fn.split("_")[-1])

    already_added = set()
    current_instances = 0
    shards_remaining = job_count
    shard_data: Deque[tuple[int, pd.DataFrame]] = deque()
    instances_per_job = get_instances_per_job(
        hours_per_job, minutes_per_job, seconds_per_instance
    )
    full_frame = pd.read_csv(input_tsv, sep="\t", low_memory=False)
    full_frame["int_study_id"] = full_frame["filename"].map(get_id_number)
    full_frame = full_frame.sort_values(by="int_study_id")
    get_frame_sbatch = partial(
        get_sbatch_script_contents, initial, job_count, hours_per_job
    )
    for raw_study_id, fn_sub_frame in full_frame.groupby(by="int_study_id"):
        study_id = cast(int, raw_study_id)
        if shards_remaining == 0:
            break
        if current_instances < instances_per_job:
            reached = current_instances + len(fn_sub_frame) == instances_per_job
            if current_instances + len(fn_sub_frame) > instances_per_job or reached:
                if (reached or len(shard_data) == 0) and study_id not in already_added:
                    shard_data.append((study_id, fn_sub_frame))
                    already_added.add(study_id)
                shard_id = (job_count - shards_remaining) + initial
                shard_dir = os.path.join(output_dir, f"shard_{shard_id}")
                mkdir(shard_dir)
                df = pd.concat(map(itemgetter(1), shard_data))
                df.drop(columns=["int_study_id"], inplace=True, axis=1)
                df.to_csv(
                    os.path.join(shard_dir, "shard_frame.tsv"), sep="\t", index=False
                )
                with open(
                    os.path.join(shard_dir, "shard_study_ids.txt"),
                    mode="w",
                    encoding="utf-8",
                ) as study_id_ls:
                    study_id_ls.write(
                        "\n".join(map(str, map(itemgetter(0), shard_data)))
                    )
                with open(
                    os.path.join(shard_dir, "shard_sbatch_job.sh"),
                    mode="w",
                    encoding="utf-8",
                ) as sbatch_script:
                    sbatch_script.write(
                        get_frame_sbatch(
                            shard_id,
                            get_buffered_hours_for_instances(
                                len(df), seconds_per_instance
                            ),
                        )
                    )
                shard_data.clear()
                full_frame.drop(df.index, inplace=True, axis=0)
                current_instances = 0
                shards_remaining -= 1
            if study_id not in already_added:
                shard_data.append((study_id, fn_sub_frame))
                already_added.add(study_id)
                current_instances += len(fn_sub_frame)
        else:
            shard_id = (job_count - shards_remaining) + initial
            shard_dir = os.path.join(output_dir, f"shard_{shard_id}")
            mkdir(shard_dir)
            df = pd.concat(map(itemgetter(1), shard_data))
            df.drop(columns=["int_study_id"], inplace=True, axis=1)
            df.to_csv(os.path.join(shard_dir, "shard_frame.tsv"), sep="\t", index=False)
            with open(
                os.path.join(shard_dir, "shard_study_ids.txt"),
                mode="w",
                encoding="utf-8",
            ) as study_id_ls:
                study_id_ls.write("\n".join(map(str, map(itemgetter(0), shard_data))))
            with open(
                os.path.join(shard_dir, "shard_sbatch_job.sh"),
                mode="w",
                encoding="utf-8",
            ) as sbatch_script:
                sbatch_script.write(
                    get_frame_sbatch(
                        shard_id,
                        get_buffered_hours_for_instances(len(df), seconds_per_instance),
                    )
                )
            shard_data.clear()
            full_frame.drop(df.index, inplace=True, axis=0)
            current_instances = 0
            shards_remaining -= 1
            # else:
            if study_id not in already_added:
                shard_data.append((study_id, fn_sub_frame))
                already_added.add(study_id)
                current_instances += len(fn_sub_frame)
    full_frame.drop(columns=["int_study_id"], inplace=True)
    full_frame.to_csv(os.path.join(output_dir, "remainder.tsv"), sep="\t", index=False)


def get_sbatch_script_contents(
    initial_shard_id: int,
    total_shards: int,
    target_hours: int,
    shard_id: int,
    estimated_hours: int,
) -> str:
    return (
        "#!/bin/bash\n"
        "#SBATCH --account=chip\n"
        "#SBATCH --partition=bch-gpu-pe             # queue to be used\n"
        f"#SBATCH --time={estimated_hours}:00:00             # Running time (in hours-minutes-seconds)\n"
        f"#SBATCH --job-name=window_crico_shard_{shard_id}           # Job name\n"
        "#SBATCH --mail-user=eli.goldner@childrens.harvard.edu      # Email address to send the job status\n"
        "#SBATCH --mail-type=END,FAIL # send and email when the job begins, ends or fails\n"
        "#SBATCH --output=/home/ch231037/logs/%x_%j.txt          # Name of the output file\n"
        "#SBATCH --nodes=1               # Number of gpu nodes\n"
        "#SBATCH --ntasks=1               # Number of gpu nodes\n"
        "#SBATCH --gres=gpu:large:1                # Number of gpu devices on one gpu node\n"
        "#SBATCH --mem=120GB\n"
        "\n"
        "source /home/ch231037/.bashrc\n"
        "source activate hf_313\n"
        "\n"
        "python ~/Repos/CRICO/inference/prompt.py  \\\n"
        "       --examples_file ~/4_core_json_vs_xml/json_first.txt \\\n"
        "       --model_path unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit \\\n"
        "       --prompt_file ~/4_core_json_vs_xml/GS_prompt_json_first.txt \\\n"
        "       --max_new_tokens 2048 \\\n"
        "       --batch_size 8 \\\n"
        "       --text_column window_text \\\n"
        f"       --query_files ~/{initial_shard_id}_{total_shards}_{target_hours}_agent_2/shard_{shard_id}/shard_frame.tsv \\\n"
        f"       --output_dir  ~/{initial_shard_id}_{total_shards}_{target_hours}_agent_2/shard_{shard_id}/processed/ \\\n"
        "       --keep_columns section_identifier	filename	medication_local_offsets	window_cas_offsets	window_text serialized_output \\\n"
    )


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

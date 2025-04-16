import argparse
import json
import logging
import os
from typing import Iterable, cast

import pandas as pd

parser = argparse.ArgumentParser(description="")

parser.add_argument("--input_tsv", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--text_column", type=str)
parser.add_argument("--counts_json", type=str)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def partition_frames(
    full_frame: pd.DataFrame, fn_to_instance_count: dict[str, int]
) -> Iterable[tuple[pd.DataFrame, bool]]:
    for fn, fn_frame in full_frame.groupby("filename"):
        fn_original_instance_count = fn_to_instance_count[cast(str, fn)]
        fn_frame_instance_count = len(fn_frame)
        if fn_frame_instance_count == fn_original_instance_count:
            yield fn_frame, True
        elif fn_frame_instance_count < fn_original_instance_count:
            logger.error(
                f"Missing instances for {fn}, {fn_frame_instance_count} in original frame {fn_original_instance_count} from cTAKES"
            )
            yield fn_frame, False
        elif fn_frame_instance_count > fn_original_instance_count:
            logger.error(
                f"INCORRECT FRAME LENGTH FOR {fn}, {fn_frame_instance_count} IN FRAME {fn_original_instance_count} FROM CTAKES"
            )
            exit(1)


def process(
    input_tsv: str, output_dir: str, text_column: str, counts_json: str
) -> None:
    with open(counts_json, mode="r", encoding="utf-8") as f:
        fn_to_instance_count = json.loads(f.read())
    full_frame = pd.read_csv(input_tsv, sep="\t", low_memory=False)
    sub_frames_ls = list(partition_frames(full_frame, fn_to_instance_count))
    finished_frame = pd.concat(
        frame for frame, is_finished in sub_frames_ls if is_finished
    )
    finished_frame.to_csv(
        os.path.join(output_dir, "finished.tsv"), sep="\t", index=False
    )
    unfinished_frame = pd.concat(
        frame for frame, is_finished in sub_frames_ls if not is_finished
    )
    unfinished_frame.to_csv(
        os.path.join(output_dir, "unfinished.tsv"), sep="\t", index=False
    )


def main() -> None:
    args = parser.parse_args()
    process(args.input_tsv, args.output_dir, args.text_column, args.counts_json)


if __name__ == "__main__":
    main()

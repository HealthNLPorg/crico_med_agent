import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_file",
    type=str,
)
parser.add_argument("--output_dir", type=str)

parser.add_argument(
    "--n_partitions",
    type=int,
)


def process(input_file: str, output_dir: str, n_partitions) -> None:
    df = pd.read_csv(input_file, sep="\t", low_memory=False)
    for index, frame in enumerate(np.array_split(df, n_partitions)):
        frame.to_csv(
            os.path.join(output_dir, f"shard_{index}.tsv"), sep="\t", index=False
        )


def main() -> None:
    args = parser.parse_args()
    process(args.input_file, args.output_dir, args.n_partitions)


if __name__ == "__main__":
    main()

import polars as pl
import os

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--tsv_dir",
    type=str,
    help="Dir full of TSVs to verify",
)

def __fast_total(tsv_path: str, separator="\t") -> int:
    lazy_frame = pl.scan_csv(tsv_path, separator=separator)
    return lazy_frame.select(pl.len()).collect().item()

def __print_totals(tsv_dir: str) -> None:
    total = 0
    for root, dirs, files in os.walk(tsv_dir):
        for fn in files:
            if fn.lower().endswith("tsv"):
                tsv_path = os.path.join(root, fn)
                fn_total = __fast_total(tsv_path)
                total += fn_total
                print(f"{fn_total}\t{tsv_path}")
    print(f"{total}\tTotal")

def main() -> None:
    args = parser.parse_args()
    __print_totals(args.tsv_dir)

if __name__ == "__main__":
    main()

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--incorrect_offset_tsv", type=str)
parser.add_argument("--correct_offset_tsv", type=str)
parser.add_argument("--output_dir", type=str)


def process(
    incorrect_offset_tsv: str, correct_offset_tsv: str, output_dir: str
) -> None:
    incorrect_offset_df = pd.read_csv(incorrect_offset_tsv, sep="\t", low_memory=False)
    correct_offset_df = pd.read_csv(correct_offset_tsv, sep="\t", low_memory=False)

    incorrect_offset_df["medication_cas_offsets"] = correct_offset_df[
        "medication_cas_offsets"
    ]
    incorrect_offset_df["window_cas_offsets"] = correct_offset_df["window_cas_offsets"]
    corrected_df = incorrect_offset_df.drop("medication_local_offsets")
    aligned_colums = {
        "section_identifier",
        "filename",
        "window_text",  # serialized_output
    }
    for column in aligned_colums:
        elementwise_comparison = incorrect_offset_df[column] == corrected_df[column]
        assert all(elementwise_comparison), (
            f"{column} items don't match {elementwise_comparison}"
        )
    corrected_df.to_csv(
        os.path.join(output_dir, os.path.basename(incorrect_offset_df)),
        sep="\t",
        index=False,
    )


def main() -> None:
    args = parser.parse_args()
    process(args.incorrect_offset_tsv, args.correct_offset_tsv, args.output_dir)


if __name__ == "__main__":
    main()

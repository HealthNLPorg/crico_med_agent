import argparse
import os

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_dir",
    type=str,
)

relevant_beginnings = {
    "*",
    "*:<span>",
    "Dosage",
    "Dosage:<span>",
    "Frequency",
    "Frequency:<span>",
    "Instruction",
    "Instruction:<span>",
    "InstructionCondition",
    "InstructionCondition:<span>",
    "Medications/Drugs",
    "Medications/Drugs:<span>",
}


def get_abbreviated_table(root: str, fn: str) -> str:
    def has_relevant_beginning(line: str) -> bool:
        return line.split()[0].strip() in relevant_beginnings

    with open(os.path.join(root, fn)) as f:
        return "\n".join(map(str.rstrip, [next(f), *filter(has_relevant_beginning, f)]))


def write_abbreviated_table(root: str, fn: str) -> None:
    abbrev_fn = "_".join(["abbrev", *fn.split("_")[1:]])
    abbreviated_table = get_abbreviated_table(root, fn)
    with open(os.path.join(root, abbrev_fn), mode="w") as f:
        f.write(abbreviated_table)


def generate_abbreviated_tables(input_dir: str) -> None:
    for root, dirs, files in os.walk(input_dir):
        for fn in files:
            if fn.startswith("full") and fn.endswith("txt"):
                write_abbreviated_table(root, fn)


def main() -> None:
    args = parser.parse_args()
    generate_abbreviated_tables(args.input_dir)


if __name__ == "__main__":
    main()

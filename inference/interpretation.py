import re
import argparse
from ast import literal_eval
from itertools import chain
from typing import Iterable, cast

import pandas as pd

from anafora_data import AnaforaDocument, Instruction, InstructionCondition, Medication

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_tsv",
    type=str,
    help="TSV with instances in the output column",
)

parser.add_argument(
    "--output_dir",
    type=str,
    help="Where to output Anafora XML",
)


def get_medication_annotation(row: pd.Series) -> Medication:
    cas_level_span = literal_eval(row["medication offsets"])
    filename = row["filename"]
    medication = Medication(span=cas_level_span, filename=filename)
    medication.set_cui_str(row["cuis"])
    medication.set_tui_str(row["tuis"])
    return medication


def to_anafora_files(corpus_frame: pd.DataFrame, output_dir: str) -> None:
    for fn, fn_frame in corpus_frame.groupby(["filename"]):
        (base_fn,) = cast(tuple[str,], fn)
        to_anafora_file(base_fn, fn_frame, output_dir)


def to_anafora_file(base_fn: str, fn_frame: pd.DataFrame, output_dir: str) -> None:
    fn_anafora_document = AnaforaDocument(filename=base_fn)
    medications: list[Medication] = fn_frame.apply(
        get_medication_annotation, axis=1
    ).to_list()
    attr_lists: list[list[Instruction | InstructionCondition]] = fn_frame.apply(
        parse_attributes, axis=1
    ).to_list()
    for medication, attr_list in zip(medications, attr_lists):
        medication.set_instructions(
            attr for attr in attr_list if isinstance(attr, Instruction)
        )
        medication.set_instruction_conditions(
            attr for attr in attr_list if isinstance(attr, InstructionCondition)
        )
    fn_anafora_document.set_entities(
        chain(
            medications,
            chain.from_iterable(attr_lists),
        )
    )
    fn_anafora_document.write_to_dir(output_dir)


def get_local_spans(
    tagged_str: str, relevant_tags: set[str]
) -> Iterable[tuple[str, int, int]]:
    tag_or_body = r"[^<>/]+"
    tag_regexes = {f"<{tag}>{tag_or_body}</{tag}>" for tag in relevant_tags}
    relevant_tags_capture = f"({'|'.join(tag_regexes)})"
    tag_and_body_capture = f"<({tag_or_body})>({tag_or_body})</{tag_or_body}>"
    current_begin = 0
    step = 0
    for run in re.split(relevant_tags_capture, tagged_str):
        potential_match = re.search(tag_and_body_capture, run)
        if potential_match is None:
            cleaned_run = re.sub(r"</?[^<>/]>", "", run)
            step = len(cleaned_run)
        else:
            tag = potential_match.group(1)
            body = potential_match.group(2)
            step = len(body)
            yield tag, current_begin, current_begin + step
        current_begin += step


def parse_attributes(row: pd.Series) -> list[Instruction | InstructionCondition]:
    filename = row["filename"]
    local_spans = get_local_spans(
        # row["output"], {"instruction", "instructionCondition"}
        row["result"],
        {"instruction", "instructionCondition"},
    )
    window_begin, _ = literal_eval(row["window offsets"])

    def to_attr(
        attr_type: str, local_begin: int, local_end: int
    ) -> Instruction | InstructionCondition:
        cas_level_span = (window_begin + local_begin, window_begin + local_end)
        return (
            Instruction(cas_level_span, filename)
            if attr_type == "instruction"
            else InstructionCondition(cas_level_span, filename)
        )

    return [
        to_attr(attr_type, local_begin, local_end)
        for attr_type, local_begin, local_end in local_spans
    ]


def output_to_result(row: pd.Series) -> str:
    full_output = literal_eval(row.output)[0]["generated_text"]
    return full_output.split("\nResult:\n")[-1].strip()


def process(input_tsv: str, output_dir: str) -> None:
    raw_frame = pd.read_csv(input_tsv, sep="\t")
    raw_frame["result"] = raw_frame.apply(output_to_result, axis=1)
    filtered_frame = raw_frame.loc[raw_frame["result"].str.lower() != "none"]
    to_anafora_files(filtered_frame, output_dir)


def main() -> None:
    args = parser.parse_args()
    process(args.input_tsv, args.output_dir)


if __name__ == "__main__":
    main()

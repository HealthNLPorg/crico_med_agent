import re
from ast import literal_eval
from itertools import chain
from typing import Iterable, cast

import pandas as pd

from anafora_data import AnaforaDocument, Instruction, InstructionCondition, Medication


def get_medication_annotation(row: pd.Series) -> Medication:
    cas_level_span = literal_eval(row["span"])
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
    tag_to_constructor = {
        "instruction": Instruction,
        "instructionCondition": InstructionCondition,
    }
    filename = row["filename"]
    local_spans = get_local_spans(
        row["output"], {"instruction", "instructionCondition"}
    )
    window_begin, _ = literal_eval(row["window offsets"])

    def to_attr(
        attr_type: str, local_begin: int, local_end: int
    ) -> Instruction | InstructionCondition:
        cas_level_span = (window_begin + local_begin, window_begin + local_end)
        return tag_to_constructor[attr_type](cas_level_span, filename)

    return [
        to_attr(attr_type, local_begin, local_end)
        for attr_type, local_begin, local_end in local_spans
    ]

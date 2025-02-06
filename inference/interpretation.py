from itertools import chain
from typing import cast
import pandas as pd
from ast import literal_eval
from .anafora_data import AnaforaDocument, Instruction, InstructionCondition, Medication


def get_local_medication_offsets(row: pd.Series) -> tuple[int, int]:
    return -1, -1


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
    instruction_lists: list[list[Instruction]] = fn_frame.apply(
        parse_instructions, axis=1
    ).to_list()
    condition_lists: list[list[InstructionCondition]] = fn_frame.apply(
        parse_conditions, axis=1
    ).to_list()
    for medication, instructions, conditions in zip(
        medications, instruction_lists, condition_lists
    ):
        medication.set_instructions(instructions)
        medication.set_instruction_conditions(conditions)
    fn_anafora_document.set_entities(
        chain(
            medications,
            chain.from_iterable(instruction_lists),
            chain.from_iterable(condition_lists),
        )
    )
    fn_anafora_document.write_to_dir(output_dir)


def parse_instructions(row: pd.Series) -> list[Instruction]:
    return []


def parse_conditions(row: pd.Series) -> list[InstructionCondition]:
    return []

import argparse
import os
import re
import json
import logging
from ast import literal_eval
from itertools import chain
from typing import cast
from collections.abc import Iterable
from operator import itemgetter
from functools import partial

import numpy as np
import pandas as pd
from anafora_data import (
    AnaforaDocument,
    Instruction,
    InstructionCondition,
    Medication,
    MedicationAttribute,
)
from utils import basename_no_ext, mkdir

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_tsv",
    type=str,
    help="TSV with instances in the output column",
)

parser.add_argument(
    "--output_dir",
    type=str,
    help="Where to output STUFF",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["anafora", "windows"],
    help="Whether stuff is Anafora XML or Windows",
)
parser.add_argument(
    "--get_differences",
    action="store_true",
    help="Log differences between JSON and XML where applicable",
)
WINDOW_RADIUS: int = 30


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


def to_anafora_file(
    base_fn: str, fn_frame: pd.DataFrame, output_dir: str, get_differences: bool
) -> None:
    fn_anafora_document = AnaforaDocument(filename=base_fn)
    medications: list[Medication] = fn_frame.apply(
        get_medication_annotation, axis=1
    ).to_list()

    fn_frame["combined"] = fn_frame["serialized_output"].map(parse_serialized_output)
    fn_frame["JSON"] = fn_frame["combined"].map(itemgetter(0))
    fn_frame["XML"] = fn_frame["combined"].map(itemgetter(1))
    fn_frame.drop(columns=["combined", "serialized_output"], inplace=True)
    fn_frame["JSON"] = fn_frame.apply(select_json, axis=1)
    fn_frame["XML"] = fn_frame.apply(select_xml, axis=1)
    attr_lists: list[list[MedicationAttribute]] = fn_frame.apply(
        parse_attributes, axis=1
    ).to_list()
    for medication, attr_list in zip(medications, attr_lists):
        medication.set_attributes(attr_list)
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
    # TODO - revisit and see if this can be replaced with re.finditer
    # though for now don't fix what isn't broken
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


# copied from ../visualization/write_to_org.py
def deserialize(s: str) -> str:
    return (
        s.replace("<cn>", "\n")
        .replace("<cr>", "\r")
        .replace("<ct>", "\t")
        .replace("<cf>", "\f")
    )


# copied from ../visualization/write_to_survey.py
def parse_serialized_output(serialized_output: str) -> tuple[list[str], list[str]]:
    model_output = deserialize(literal_eval(serialized_output)[0])
    # print(re.split(r"(XML\:\s*[^\{\}]*|JSON\:\s*\{[^\{\}\}]*\})", model_output))
    if model_output.strip().lower() == "none":
        return ["None"], ["None"]
    groups = re.split(r"(XML\:\s*[^\{\}]*|JSON\:\s*\{[^\{\}\}]*\})", model_output)
    json_raw_parses = [
        parse_group[5:].strip()
        for parse_group in groups
        if parse_group.strip().lower().startswith("json:")
    ]
    xml_raw_parses = [
        parse_group[4:].strip()
        for parse_group in groups
        if parse_group.strip().lower().startswith("xml:")
    ]
    return json_raw_parses, xml_raw_parses


def get_tag_body(window_text: str, xml_tag: str) -> str:
    matches = re.findall(rf"<{xml_tag}>(.+)</{xml_tag}>", window_text)
    if len(matches) == 0:
        # logger.error(f"Window with no real medications:\n\n{window_text}")
        return ""
    return matches[0]


get_medication = partial(get_tag_body, xml_tag="medication")


def select_json(row: pd.Series) -> str:
    for json_str in row["JSON"]:
        try:
            raw_dict = json.loads(json_str)
        except Exception:
            continue
        json_med_str = " , ".join(raw_dict.get("medication", [])).strip().lower()
        if json_med_str == row["medication"].strip().lower():
            return json_str
    return ""


def select_xml(row: pd.Series) -> str:
    for xml_str in row["XML"]:
        if get_medication(xml_str).strip().lower() == row["medication"].strip().lower():
            return xml_str
    return ""


def json_str_to_dict(json_str: str) -> dict[str, str]:
    relevant_attributes = {"dosage", "frequency", "instruction", "condition"}

    def _normalize_value(raw_json_dict: dict[str, list[str]], key: str) -> str:
        raw_result = raw_json_dict.get(key, default=[""])[0]
        return " ".join(raw_result.strip().lower().split())

    try:
        raw_json_dict = json.loads(json_str)
        normalize_value = partial(_normalize_value, raw_json_dict=raw_json_dict)
        return {
            relevant_attribute: normalize_value(relevant_attribute)
            for relevant_attribute in relevant_attributes
        }
    except Exception as _:
        logger.warning(f"Could not parse JSON string: {json_str}")
        return dict()


def parse_attributes(row: pd.Series) -> list[MedicationAttribute]:
    filename = row["filename"]
    if row["result"].lower() == "none":
        return []
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


def anafora_process(
    input_tsv: str, output_dir: str, get_differences: bool = False
) -> None:
    raw_frame = pd.read_csv(input_tsv, sep="\t")
    raw_frame["result"] = raw_frame.apply(output_to_result, axis=1)
    # filtered_frame = raw_frame.loc[raw_frame["result"].str.lower() != "none"]
    # to_anafora_files(filtered_frame, output_dir)
    to_anafora_files(raw_frame, output_dir, get_differences)


def build_frame_with_med_windows(raw_frame: pd.DataFrame) -> pd.DataFrame:
    def serialized_output_to_unique_meds(output: list[str]) -> set[str]:
        raw_output = re.sub("<c[rtnf]>", "", output[0])
        raw_output = re.sub("\(", "\\(", raw_output)
        raw_output = re.sub("\)", "\\)", raw_output)
        raw_output = re.sub("\[", "\\[", raw_output)
        raw_output = re.sub("\]", "\\]", raw_output)
        raw_output = re.sub("\+", "\\+", raw_output)
        normalized = raw_output.lower()
        bad_terms = {
            "begin_of_text",
            "end_of_text",
            "start_header_id",
            "end_header_id",
            "eot_id",
            ":",
        }
        if normalized == "none" or any(
            bad_term in normalized for bad_term in bad_terms
        ):
            return {}
        return {med.lower().strip() for med in raw_output.split(",")}

    def get_central_index(med_begin: int, token_index_ls: list[tuple[int, int]]) -> int:
        def closest(token_ord: int) -> int:
            return abs(token_index_ls[token_ord][0] - med_begin)

        return min(range(len(token_index_ls)), default=-1, key=closest)

    def build_med_windows(
        section_body: str, meds: set[str]
    ) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
        meds_regex = "|".join(meds)
        normalized_section = section_body.lower()
        token_index_ls = [token.span() for token in re.finditer(r"\S+", section_body)]

        def match_to_window(med_match) -> tuple[tuple[int, int], tuple[int, int], str]:
            med_begin, med_end = med_match.span()
            med_central_index = get_central_index(med_begin, token_index_ls)
            window_begin = token_index_ls[max(0, med_central_index - WINDOW_RADIUS)][0]
            window_end = token_index_ls[
                min(len(token_index_ls) - 1, med_central_index + WINDOW_RADIUS)
            ][1]
            opening = normalized_section[window_begin:med_begin]
            tagged_medication = normalized_section[med_begin:med_end]
            closing = normalized_section[med_end:window_end]
            window = (
                f"...{opening}<medication>{tagged_medication}</medication>{closing}..."
            )
            return ((med_begin, med_end), (window_begin, window_end), window)

        return [
            match_to_window(med_match)
            for med_match in re.finditer(meds_regex, normalized_section)
        ]

    def row_to_window_list(
        row: pd.Series,
    ) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
        meds = serialized_output_to_unique_meds(literal_eval(row.serialized_output))
        if (
            len(meds) == 0
            or (isinstance(row.section_body, str) and len(row.section_body) == 0)
            or (
                not isinstance(row.section_body, str)
                and (row.section_body is None or np.isnan(row.section_body))
            )
        ):
            return []
        return build_med_windows(str(row.section_body), meds)

    raw_frame["raw_windows"] = raw_frame.apply(row_to_window_list, axis=1)
    raw_frame = raw_frame[raw_frame["raw_windows"].astype(bool)]
    full_frame = raw_frame.explode("raw_windows")

    def get_window_med_local_offsets(row: pd.Series) -> tuple[int, int]:
        return row.raw_windows[0]

    def get_window_cas_offsets(row: pd.Series) -> tuple[int, int]:
        return row.raw_windows[1]

    def get_window_text(row: pd.Series) -> str:
        return row.raw_windows[2]

    full_frame["medication_local_offsets"] = full_frame.apply(
        get_window_med_local_offsets, axis=1
    )
    full_frame["window_cas_offsets"] = full_frame.apply(get_window_cas_offsets, axis=1)
    full_frame["window_text"] = full_frame.apply(get_window_text, axis=1)
    full_frame.drop("raw_windows", axis=1, inplace=True)
    full_frame.reset_index(drop=True)
    return full_frame


def windows_process(input_tsv: str, output_dir: str) -> None:
    raw_frame = pd.read_csv(input_tsv, sep="\t")
    expanded_windows_frame = build_frame_with_med_windows(raw_frame)
    mkdir(output_dir)
    input_file_basename = basename_no_ext(input_tsv)
    out_path = os.path.join(output_dir, f"windowed_{input_file_basename}.tsv")
    expanded_windows_frame.to_csv(out_path, sep="\t", index=False)


def main() -> None:
    args = parser.parse_args()
    match args.mode:
        case "anafora":
            anafora_process(args.input_tsv, args.output_dir, args.get_differences)
        case "windows":
            windows_process(args.input_tsv, args.output_dir)


if __name__ == "__main__":
    main()

import argparse
import os
import re
import json
import logging
from ast import literal_eval
from itertools import chain, islice
from typing import cast, Any
from collections.abc import Iterable
from collections import Counter
from operator import itemgetter
from functools import partial

import numpy as np
import pandas as pd
from anafora_data import (
    AnaforaDocument,
    Instruction,
    InstructionCondition,
    Dosage,
    Frequency,
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


class ConfusionMatrix(Counter):
    def __init__(self, tp_total: int, fp_total: int, fn_total: int) -> None:
        self.totals = Counter(
            {
                "TP": tp_total,
                "FP": fp_total,
                "FN": fn_total,
            }
        )

    def is_complete_hallucination(self) -> bool:
        return self.totals["FP"] > 0 and self.totals["TP"] == 0


def get_medication_anafora_annotation(row: pd.Series) -> Medication:
    # inappropriately named because I SCREWED UP
    # YES THAT'S RIGHT I SCREWED UP CAN YOU BELIEVE IT???????????
    cas_level_span = literal_eval(row["medication_local_offsets"])
    filename = row["filename"]
    medication = Medication(span=cas_level_span, filename=filename)
    medication.set_cui_str(row.get("cuis", ""))
    medication.set_tui_str(row.get("tuis", ""))
    return medication


def get_medication_text(window_text: str) -> str | None:
    matches = get_tagged_bodies("medication", window_text)
    assert len(matches) == 1 or len(matches) == 0, window_text
    if len(matches) == 0:
        return None
    return matches[0]


def to_anafora_files(
    corpus_frame: pd.DataFrame, output_dir: str, get_differences: bool
) -> None:
    for fn, fn_frame in corpus_frame.groupby(["filename"]):
        (base_fn,) = cast(tuple[str,], fn)
        to_anafora_file(base_fn, fn_frame, output_dir, get_differences)


def to_anafora_file(
    base_fn: str, fn_frame: pd.DataFrame, output_dir: str, get_differences: bool
) -> None:
    fn_anafora_document = AnaforaDocument(filename=base_fn)
    medications: list[Medication] = fn_frame.apply(
        get_medication_anafora_annotation, axis=1
    ).to_list()

    def contains_tags(tag_core: str, target: str) -> bool:
        open_tag = f"<{tag_core}>"
        close_tag = f"</{tag_core}>"
        return open_tag in target and close_tag in target

    contains_med_tags = partial(contains_tags, "medication")
    fn_frame = fn_frame.loc[fn_frame["window_text"].map(contains_med_tags)]
    fn_frame["medication"] = fn_frame["window_text"].map(get_medication_text)
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


def get_local_spans_from_xml(
    tagged_str: str, relevant_tags: set[str]
) -> Iterable[tuple[str, int, int]]:
    tag_or_body = r"[^<>/]+"
    tag_regexes = {rf"<{tag}>{tag_or_body}</{tag}>" for tag in relevant_tags}
    relevant_tags_capture = rf"({'|'.join(tag_regexes)})"
    tag_and_body_capture = rf"<({tag_or_body})>({tag_or_body})</{tag_or_body}>"
    current_begin = 0
    step = 0
    # TODO - revisit and see if this can be replaced with re.finditer
    # though for now don't fix what isn't broken
    # TODO - might have to empirically
    # confirm this via the anafora visualization
    # code in format-writer
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


def get_local_spans_from_json(
    window_text: str, json_dict: dict[str, Counter[str]]
) -> Iterable[tuple[str, int, int]]:
    def get_span(re_match_obj: re.Match) -> tuple[int, int]:
        return re_match_obj.span()

    def get_matches(
        occurence_count: Counter[str], window_text: str
    ) -> Iterable[tuple[int, int]]:
        for occurence, count in occurence_count.items():
            all_matches = re.finditer(occurence, window_text)
            yield from strategy(all_matches, count)

    def strategy(matches: Iterable[re.Match[str]], count) -> Iterable[tuple[int, int]]:
        # TODO tweak as necessary - might have to add other information from the row
        return map(get_span, islice(matches, count))

    # TODO adapt to cases with multiple values
    for attr, occurence_count in json_dict.items():
        # for match_span in strategy(map(get_span, re.finditer(body, window_text))):
        for match_span in get_matches(occurence_count, window_text):
            begin, end = match_span
            yield attr, begin, end


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


def get_tagged_bodies(xml_tag: str, window_text: str) -> list[str]:
    return re.findall(rf"<{xml_tag}>(.+)</{xml_tag}>", window_text)


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
        if (
            get_medication_text(xml_str) is not None
            # mypy doesn't like this but the first clause will short circuit
            # if the result isn't a string
            and get_medication_text(xml_str).strip().lower()
            == row["medication"].strip().lower()
        ):
            return xml_str
    return ""


def json_str_to_dict(json_str: str) -> dict[str, Counter[str]]:
    relevant_attributes = {"dosage", "frequency", "instruction", "condition"}
    # TODO - figure out if the one getting lost in the test set is
    # actually a bug
    if len(json_str.strip()) == 0:
        return dict()
    try:
        raw_json_dict = json.loads(json_str)
    except Exception as e:
        logger.warning(f"Could not parse JSON string: {json_str} - from exception {e}")
        return dict()
    return {
        relevant_attribute: Counter(raw_json_dict.get(relevant_attribute, []))
        for relevant_attribute in relevant_attributes
    }


def xml_str_to_dict(xml_str: str) -> dict[str, Counter[str]]:
    relevant_attributes = {"dosage", "frequency", "instruction", "condition"}

    return {
        relevant_attribute: Counter(get_tagged_bodies(relevant_attribute, xml_str))
        for relevant_attribute in relevant_attributes
    }


def get_cf_dict(
    attr_name_to_instances: dict[str, Counter[str]], ground_truth: str
) -> dict[str, dict[str, ConfusionMatrix]]:
    def compute_hallucinations(
        disovered_instances: Counter[str],
    ) -> dict[str, ConfusionMatrix]:
        instance_to_cfm = {}
        for instance, prediction_count in disovered_instances.items():
            gold_count = sum(
                1 for _ in re.findall(re.escape(instance), re.escape(ground_truth))
            )
            # rough justice since not span level but
            if gold_count == prediction_count:
                instance_to_cfm[instance] = ConfusionMatrix(
                    tp_total=gold_count, fp_total=0, fn_total=0
                )
            elif gold_count > prediction_count:
                instance_to_cfm[instance] = ConfusionMatrix(
                    tp_total=prediction_count,
                    fp_total=0,
                    fn_total=gold_count - prediction_count,
                )
            else:
                instance_to_cfm[instance] = ConfusionMatrix(
                    tp_total=prediction_count,
                    fp_total=prediction_count - gold_count,
                    fn_total=0,
                )
        return instance_to_cfm

    return {
        attr: compute_hallucinations(instances)
        for attr, instances in attr_name_to_instances.items()
    }


def parse_attributes(row: pd.Series) -> list[MedicationAttribute]:
    if row["JSON"] == "" and row["XML"] == "":
        return []
    json_dict = json_str_to_dict(row["JSON"])
    xml_dict = xml_str_to_dict(row["XML"])
    xml_cf_dict = get_cf_dict(xml_dict, row["window_text"])
    json_cf_dict = get_cf_dict(json_dict, row["window_text"])

    def parse_has_no_total_hallucinations(
        cf_dict: dict[str, dict[str, ConfusionMatrix]],
    ) -> bool:
        return not any(
            cf.is_complete_hallucination()
            for instance_to_cf in cf_dict.values()
            for cf in instance_to_cf.values()
        )

    def parse_is_all_total_hallucinations(
        cf_dict: dict[str, dict[str, ConfusionMatrix]],
    ) -> bool:
        return all(
            cf.is_complete_hallucination()
            for instance_to_cf in cf_dict.values()
            for cf in instance_to_cf.values()
        )

    def filter_hallucinatory(
        occurence_dict: dict[str, Counter[str]],
        cf_dict: dict[str, dict[str, ConfusionMatrix]],
    ) -> dict[str, Counter[str]]:
        # return {
        #     k: v for k, v in occurence_dict.items() if not json_cf_dict.get(k, False)
        # }
        def compare(
            ocurrence_count: Counter[str], occurence_to_cf: dict[str, ConfusionMatrix]
        ) -> Counter[str]:
            filtered_count: Counter[str] = Counter()
            for occurence, count in occurence_count.items():
                cf = occurence_to_cf[occurence]
                if not cf.is_complete_hallucination():
                    # filtered_count[occurence] = count - cf.totals["FP"]
                    # keep it true to what the model predicted
                    filtered_count[occurence] = count
            return filtered_count

        filtered_occurence_dict: dict[str, Counter[str]] = {}
        for attr, occurence_count in occurence_dict.items():
            filtered_occurence_count = compare(occurence_count, cf_dict[attr])
            if len(filtered_occurence_count) > 0:
                filtered_occurence_dict[attr] = filtered_occurence_count
        return filtered_occurence_dict

    if json_dict == xml_dict:
        if parse_has_no_total_hallucinations(
            xml_cf_dict
        ) and parse_has_no_total_hallucinations(json_cf_dict):
            logger.info("JSON and XML agree and are entirely non-hallucinatory")
            return get_spans_from_xml(
                row, {"instruction", "condition", "dosage", "frequency"}
            )
        else:
            assert xml_cf_dict == json_cf_dict, (
                f"Disagreement at hallucination level but not text level for JSON {json_dict} and XML {xml_dict}"
            )
            if all(xml_cf_dict.values()):
                logger.info(
                    "JSON and XML agree and are both entirely hallucinatory - dumping instance"
                )
                return []
            else:
                logger.info(
                    "JSON and XML agree and are partially hallucinatory - defaulting to non-hallucinatory JSON for cleaner parsing"
                )
                return get_spans_from_json(
                    row,
                    filter_hallucinatory(
                        occurence_dict=json_dict, cf_dict=json_cf_dict
                    ),
                )
    else:
        if parse_has_no_total_hallucinations(xml_cf_dict):
            logger.info("JSON and XML disagree - XML is non-hallucinatory")
            return get_spans_from_xml(
                row, {"instruction", "condition", "dosage", "frequency"}
            )
        elif parse_is_all_total_hallucinations(
            xml_cf_dict
        ) and parse_is_all_total_hallucinations(json_cf_dict):
            logger.info(
                "JSON and XML disagree and are both entirely hallucinatory - dumping instance"
            )
            return []
        else:
            logger.info(
                "JSON and XML disagree and XML is partially hallucinatory - defaulting to non-hallucinatory JSON for cleaner parsing"
            )
            return get_spans_from_json(
                row,
                filter_hallucinatory(occurence_dict=json_dict, cf_dict=json_cf_dict),
            )


def build_medication_attribute(
    filename: str, window_begin: int, attr_type: str, local_begin: int, local_end: int
) -> MedicationAttribute | None:
    cas_level_span = (window_begin + local_begin, window_begin + local_end)
    match attr_type:
        case "instruction":
            return Instruction(cas_level_span, filename)
        case "condition":
            return InstructionCondition(cas_level_span, filename)
        case "dosage":
            return Dosage(cas_level_span, filename)
        case "frequency":
            return Frequency(cas_level_span, filename)
        case other:
            if other != "medication":
                logger.warning(
                    f"Ignoring tag {other} - if medication it is because we parsed it earlier"
                )
            return None


def get_spans_from_xml(row: pd.Series, attrs: set[str]) -> list[MedicationAttribute]:
    filename = row["filename"]
    local_spans = get_local_spans_from_xml(
        # row["result"],
        row["XML"],
        # {"instruction", "instructionCondition"},
        attrs,
    )
    window_begin, _ = literal_eval(row["window_cas_offsets"])

    to_attr = partial(build_medication_attribute, filename, window_begin)
    result = cast(
        list[MedicationAttribute],
        [
            to_attr(attr_type, local_begin, local_end)
            for attr_type, local_begin, local_end in local_spans
            if to_attr(attr_type, local_begin, local_end) is not None
        ],
    )
    return result


def get_spans_from_json(
    row: pd.Series,
    json_dict: dict[str, Counter[str]],
) -> list[MedicationAttribute]:
    # some possible options,
    # we can find all the match spans using re.finditer,
    # then can choose one or more spans
    # by some strategy, e.g. first,
    # last, closest to the medication.
    # How often this is necessary and
    # what strategy to use should be determined on the dev set
    filename = row["filename"]
    local_spans = get_local_spans_from_json(
        row["window_text"],
        json_dict,
    )
    window_begin, _ = literal_eval(row["window offsets"])
    to_attr = partial(build_medication_attribute, filename, window_begin)

    result = cast(
        list[MedicationAttribute],
        [
            to_attr(attr_type, local_begin, local_end)
            for attr_type, local_begin, local_end in local_spans
            if to_attr(attr_type, local_begin, local_end) is not None
        ],
    )
    return result


def output_to_result(row: pd.Series) -> str:
    full_output = literal_eval(row.output)[0]["generated_text"]
    return full_output.split("\nResult:\n")[-1].strip()


def anafora_process(
    input_tsv: str, output_dir: str, get_differences: bool = False
) -> None:
    raw_frame = pd.read_csv(input_tsv, sep="\t")
    # raw_frame["result"] = raw_frame.apply(output_to_result, axis=1)
    # filtered_frame = raw_frame.loc[raw_frame["result"].str.lower() != "none"]
    # to_anafora_files(filtered_frame, output_dir)
    to_anafora_files(raw_frame, output_dir, get_differences)


def build_frame_with_med_windows(raw_frame: pd.DataFrame) -> pd.DataFrame:
    def serialized_output_to_unique_meds(output: list[str]) -> set[str]:
        raw_output = re.sub("<c[rtnf]>", "", output[0])
        normalized = re.escape(raw_output).lower()
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
            return set()
        return {med.lower().strip() for med in raw_output.split(",") if len(med.strip()) > 0}

    def get_central_index(med_begin: int, token_index_ls: list[tuple[int, int]]) -> int:
        def closest(token_ord: int) -> int:
            return abs(token_index_ls[token_ord][0] - med_begin)

        return min(range(len(token_index_ls)), default=-1, key=closest)

    def build_med_windows(
        section_body: str, meds: set[str]
    ) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
        meds_regex = "|".join(map(re.escape, meds))
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
    def non_empty(s :list[Any]) -> bool:
        return len(s) > 0
    raw_frame[~raw_frame["raw_windows"].map(non_empty)][["serialized_output", "section_body"]].to_csv("/home/etg/Repos/CRICO/testing_escape/problem_children.tsv", sep="\t")
    raw_frame = raw_frame[raw_frame["raw_windows"].map(non_empty)]
    full_frame = raw_frame.explode("raw_windows")

    full_frame["medication_local_offsets"] = full_frame["raw_windows"].map(
        itemgetter(0)
    )

    full_frame["window_cas_offsets"] = full_frame["raw_windows"].map(itemgetter(1))

    full_frame["window_text"] = full_frame["raw_windows"].map(itemgetter(2))

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

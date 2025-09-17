import argparse
import json
import logging
import os
import re
from ast import literal_eval
from collections import Counter
from collections.abc import Iterable
from functools import partial
from itertools import chain, groupby, islice
from operator import attrgetter, itemgetter
from typing import Any, cast

import numpy as np
import pandas as pd
from lxml import etree
from lxml.etree import (
    _Element,
)

from ..utils import basename_no_ext, mkdir, parse_serialized_output
from .anafora_data import (
    AnaforaDocument,
    Dosage,
    Frequency,
    Instruction,
    InstructionCondition,
    Medication,
    MedicationAttribute,
)

DATA_SPACES = ["anafora_xml", "agent_1_output", "agent_2_output", "json_lines"]
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
    default=None,
)

parser.add_argument(
    "--paired_anafora_dir",
    type=str,
    help="TSV with instances in the output column",
    default=None,
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Where to output STUFF",
    default=None,
)
parser.add_argument(
    "--input_mode",
    type=str,
    choices=DATA_SPACES,
    help="Representation space of the input data",
)
parser.add_argument(
    "--output_mode",
    type=str,
    choices=DATA_SPACES,
    help="Target representation space",
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


# In [50]: import json
# In [51]: json.dumps({"_json_true": True, "_json_false_": False})
# Out[51]: '{"_json_true": true, "_json_false_": false}'
# It gets taken care of
def __file_to_unmerged_dictionaries(
    xml_path: str, note_path: str
) -> Iterable[dict[str, str | int | bool]]:
    with open(xml_path, mode="rb") as xml_f:
        anafora_xml = etree.fromstring(xml_f.read())

    with open(note_path) as note_f:
        note_text = note_f.read()

    __local_med_dict = partial(
        __build_medication_dictionary_from_anafora,
        note_text,
    )
    for annotation in anafora_xml.find("annotations"):
        if (
            annotation.tag == "entity"
            and annotation.find("type").text == "Medications/Drugs"
        ):
            yield __local_med_dict(annotation)


def __file_to_merged_dictionaries(
    study_id: int, xml_path: str, note_path: str
) -> Iterable[dict[str, str | int | bool]]:
    unmerged_medication_dictionaries = sorted(
        __file_to_unmerged_dictionaries(xml_path, note_path),
        key=itemgetter("medication"),
    )
    for medication, medication_dictionaries_iter in groupby(
        unmerged_medication_dictionaries,
        key=itemgetter("medication"),
    ):
        # to avoid consumption on repeat iterations
        medication_dictionaries = list(medication_dictionaries_iter)
        yield {
            "study_id": study_id,
            "medication": medication,
            "has_at_least_one_instruction": any(
                map(itemgetter("has_at_least_one_instruction"), medication_dictionaries)
            ),
            "has_at_least_one_condition": any(
                map(itemgetter("has_at_least_one_condition"), medication_dictionaries)
            ),
        }


def __build_medication_dictionary_from_anafora(
    note_text: str,
    medication_annotation: _Element,
) -> dict[str, str | int | bool]:
    begin, end = [
        int(idx) for idx in medication_annotation.find("span").text.split(",")
    ]
    properties = medication_annotation.find("properties")
    return {
        "medication": note_text[begin:end].strip().lower(),
        "has_at_least_one_instruction": any(
            p.text for p in properties.findall("instruction_")
        ),
        "has_at_least_one_condition": any(
            p.text for p in properties.findall("instruction_condition")
        ),
    }


def __get_med_json_line(medication_dictionary: dict[str, str | int | bool]) -> str:
    return f"{json.dumps(medication_dictionary)}\n"


def __dir_to_dictionaries(
    anafora_dir: str,
) -> Iterable[dict[str, str | int | bool]]:
    def get_study_id(subdir: str) -> int:
        return int(subdir.split("_")[-1])

    def get_xml_and_note_fns(abs_subdir: str) -> tuple[str | None, str]:
        relevant_files = (fn for fn in os.listdir(abs_subdir) if fn.startswith("Study"))
        relevant_files_sorted = sorted(relevant_files, reverse=True)
        if len(relevant_files_sorted) == 2:
            xml_fn, note_fn = relevant_files_sorted
            return xml_fn, note_fn
        elif len(relevant_files_sorted) == 1:
            return None, relevant_files_sorted[0]
        else:
            raise ValueError(f"{abs_subdir} has invalid contents: {relevant_files}")

    for subdir in os.listdir(anafora_dir):
        if subdir.startswith("Study"):
            study_id = get_study_id(subdir)
            abs_subdir = os.path.join(anafora_dir, subdir)
            xml_fn, note_fn = get_xml_and_note_fns(abs_subdir)
            if xml_fn is not None:
                xml_path = os.path.join(abs_subdir, xml_fn)
                note_path = os.path.join(abs_subdir, note_fn)
                yield from __file_to_merged_dictionaries(
                    study_id=study_id, xml_path=xml_path, note_path=note_path
                )


def agent_2_to_json_lines(
    input_tsv: str, output_dir: str, get_differences: bool = False
) -> None:
    corpus_frame = pd.read_csv(input_tsv, sep="\t").fillna("")

    out_path = os.path.join(output_dir, f"{os.path.basename(input_tsv)}.jsonl")

    with open(out_path, mode="w", encoding="utf-8") as f:
        for fn, fn_frame in corpus_frame.groupby(["filename"]):
            (base_fn,) = cast(tuple[str,], fn)
            for (
                medication_dictionary
            ) in __build_medication_dictionaries_from_file_frame(
                base_fn, fn_frame, output_dir, get_differences
            ):
                f.write(__get_med_json_line(medication_dictionary))


def __build_medication_dictionaries_from_file_frame(
    base_fn: str, fn_frame: pd.DataFrame, output_dir: str, get_differences: bool
) -> Iterable[dict[str, str | int | bool]]:
    medications, _ = get_medications_aligned_with_attributes(fn_frame)
    study_id_number = int(base_fn.split("_")[-1])

    def __normalize_med_text(medication: Medication) -> str:
        return medication.get_text().strip().lower()

    sorted_medications = sorted(medications, key=__normalize_med_text)

    def __cluster_has_at_least_one_of_attr_name(
        med_cluster_ls: list[Medication], attr_name: str
    ) -> bool:
        return any(map(len, map(attrgetter(attr_name), med_cluster_ls)))

    for normalized_med_text, same_med_cluster_iter in groupby(
        sorted_medications, key=__normalize_med_text
    ):
        same_med_cluster_ls = list(same_med_cluster_iter)
        # NB, these keys don't match those in
        # build_medication_attribute (currently line 600),
        # but that's bc of the attribute
        # names in the Medication class
        # in ./anafora_data.py
        if study_id_number == 121_379 and normalized_med_text == "omeprazole":
            for med in same_med_cluster_ls:
                print(str(med))
        yield {
            "study_id": study_id_number,
            "medication": normalized_med_text,
            "has_at_least_one_instruction": __cluster_has_at_least_one_of_attr_name(
                same_med_cluster_ls, "instructions"
            ),
            "has_at_least_one_condition": __cluster_has_at_least_one_of_attr_name(
                same_med_cluster_ls, "instruction_conditions"
            ),
        }


def anafora_to_json_lines(
    paired_anafora_dir: str,
    output_dir: str,
) -> None:
    # str.rstrip since if it ends with /
    # basename returns ""
    base_folder_name = os.path.basename(paired_anafora_dir.rstrip("/"))
    out_path = os.path.join(output_dir, f"{base_folder_name}.jsonl")

    with open(out_path, mode="w", encoding="utf-8") as f:
        for medication_dictionary in __dir_to_dictionaries(paired_anafora_dir):
            f.write(__get_med_json_line(medication_dictionary))


def get_medication_anafora_annotation(row: pd.Series) -> Medication:
    medication_cas_offsets = literal_eval(row["medication_cas_offsets"])
    filename = row["filename"]
    medication = Medication(
        span=medication_cas_offsets, filename=filename, text=row["medication"]
    )
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


def contains_tags(tag_core: str, target: str) -> bool:
    open_tag = f"<{tag_core}>"
    close_tag = f"</{tag_core}>"
    return open_tag in target and close_tag in target


def retain_subframe_with_validated_windows(
    fn_frame: pd.DataFrame,
) -> pd.DataFrame:
    contains_med_tags = partial(contains_tags, "medication")
    fn_frame = fn_frame.loc[fn_frame["window_text"].map(contains_med_tags)]
    return fn_frame


def unfold_frame(
    fn_frame: pd.DataFrame,
) -> pd.DataFrame:
    fn_frame["medication"] = fn_frame["window_text"].map(get_medication_text)
    fn_frame["combined"] = fn_frame["serialized_output"].map(parse_serialized_output)
    fn_frame["JSON"] = fn_frame["combined"].map(itemgetter(0))
    fn_frame["XML"] = fn_frame["combined"].map(itemgetter(1))
    fn_frame.drop(columns=["combined", "serialized_output"], inplace=True)
    fn_frame["JSON"] = fn_frame.apply(select_json, axis=1)
    fn_frame["XML"] = fn_frame.apply(select_xml, axis=1)
    return fn_frame


def get_medications_aligned_with_attributes(
    fn_frame: pd.DataFrame,
) -> tuple[list[Medication], list[MedicationAttribute]]:
    fn_frame = retain_subframe_with_validated_windows(fn_frame)
    fn_frame = unfold_frame(fn_frame)
    medications: list[Medication] = fn_frame.apply(
        get_medication_anafora_annotation, axis=1
    ).to_list()
    attr_lists: list[list[MedicationAttribute]] = fn_frame.apply(
        parse_attributes, axis=1
    ).to_list()
    for medication, attr_list in zip(medications, attr_lists):
        medication.set_attributes(attr_list)
    return medications, attr_lists


def to_anafora_file(
    base_fn: str, fn_frame: pd.DataFrame, output_dir: str, get_differences: bool
) -> None:
    medications, attr_lists = get_medications_aligned_with_attributes(fn_frame)
    fn_anafora_document = AnaforaDocument(filename=base_fn)
    fn_anafora_document.set_entities(
        chain(
            medications,
            chain.from_iterable(attr_lists),
        )
    )
    fn_anafora_document.write_to_dir(output_dir)


# DONE - check span adjustment logic
def get_local_spans_from_xml(
    tagged_str: str, relevant_tags: set[str], debug_print=True
) -> Iterable[tuple[str, int, int]]:
    # tag_or_body = r"[^<>/]+"
    # DONE - current best solution according to metrics and
    # parsing behavior from lxml
    tag_or_body = r"[^<>]+"
    # tag_or_body = r"[^<]+"
    # tag_or_body = r".+"
    tag_regexes = {rf"<{tag}>{tag_or_body}</{tag}>" for tag in relevant_tags}
    relevant_tags_capture = rf"({'|'.join(tag_regexes)})"
    tag_and_body_capture = rf"<({tag_or_body})>({tag_or_body})</{tag_or_body}>"
    current_begin = 0
    step = 0
    # TODO - revisit and see if this can be replaced with re.finditer
    # though for now don't fix what isn't broken
    # DONE - might have to empirically
    # confirm this via the anafora visualization
    # code in format-writer
    # COMMENT found the explanation
    for run in re.split(relevant_tags_capture, tagged_str):
        potential_match = re.search(tag_and_body_capture, run)
        if potential_match is None:
            # TODO this might need tweaking (probably not)
            # But if running into issues with spans
            # try r"</?[^<>]>"
            # (difference is the internal tag regex matches the earlier one)
            cleaned_run = re.sub(r"</?[^<>/]>", "", run)
            step = len(cleaned_run)
        else:
            tag = potential_match.group(1)
            body = potential_match.group(2)
            if debug_print:
                # TODO - need condition for the example
                logger.info(f"MATCHED - {tag} - {body}")
            step = len(body)
            yield tag, current_begin, current_begin + step
        current_begin += step


# DONE - check span adjustment logic
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
        # DONE tweak as necessary - might have to add other information from the row
        # COMMENT will leave this as it is since the idea might be
        # useful later even if not for this project
        return map(get_span, islice(matches, count))

    # DONE adapt to cases with multiple values
    # COMMENT not sure what I meant by the above comment,
    # but I think this already works how it's supposed to?
    for attr, occurence_count in json_dict.items():
        # for match_span in strategy(map(get_span, re.finditer(body, window_text))):
        for match_span in get_matches(occurence_count, window_text):
            begin, end = match_span
            yield attr, begin, end


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
    # TODO figure out which instance above was the issue
    # finding out when this question was inserted into the code was:
    # (base) etg@laptop:~/Repos/CRICO/src/crico/evaluation$ git log -S "# TODO - figure out if the one getting lost in the test set is"
    # commit 20c866e90b1045096388fb58a35361bd70c12f79
    # Author: Eli Goldner <etgld@posteo.us>
    # Date:   Fri May 9 22:46:15 2025 -0400
    # Working on both dev and test - time to evaluate
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


def parse_attributes(row: pd.Series) -> list[MedicationAttribute]:
    if row["JSON"] == "" and row["XML"] == "":
        return []
    json_dict = json_str_to_dict(row["JSON"])
    xml_dict = xml_str_to_dict(row["XML"])
    xml_cf_dict = get_cf_dict(xml_dict, row["window_text"])
    json_cf_dict = get_cf_dict(json_dict, row["window_text"])

    if json_dict == xml_dict:
        if parse_has_no_total_hallucinations(
            xml_cf_dict
        ) and parse_has_no_total_hallucinations(json_cf_dict):
            logger.info("JSON and XML agree and are entirely non-hallucinatory")
            return get_medication_attributes_from_xml(
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
                return get_medication_attributes_from_json(
                    row,
                    filter_hallucinatory(
                        occurence_dict=json_dict, cf_dict=json_cf_dict
                    ),
                )
    else:
        if parse_has_no_total_hallucinations(xml_cf_dict):
            logger.info("JSON and XML disagree - XML is non-hallucinatory")
            med_attrs = get_medication_attributes_from_xml(
                row, {"instruction", "condition", "dosage", "frequency"}
            )
            # if row["filename"] == "Study_ID_121379" and row["medication"] == "omeprazole":
            #     print(row)
            #     print(row["JSON"])
            #     print(row["XML"])
            #     for med_attr in med_attrs:
            #         print(str(med_attr))
            return med_attrs
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
            return get_medication_attributes_from_json(
                row,
                filter_hallucinatory(occurence_dict=json_dict, cf_dict=json_cf_dict),
            )


# DONE - check span adjustment logic
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


# DONE - check span adjustment logic
def get_medication_attributes_from_xml(
    row: pd.Series, attrs: set[str]
) -> list[MedicationAttribute]:
    filename = row["filename"]
    debug_span = False
    # if row["filename"] == "Study_ID_121379" and row["medication"] in {
    #     "omeprazole",
    #     "carafate",
    #     "avastin",
    # }:
    #     print(row)
    #     print(row["JSON"])
    #     print(row["XML"])
    #     debug_span = True
    local_spans = get_local_spans_from_xml(
        # row["result"],
        row["XML"],
        # {"instruction", "instructionCondition"},
        attrs,
        # debug_print=debug_span,
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


# DONE - check span adjustment logic
def get_medication_attributes_from_json(
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


def agent_2_to_anafora(
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
        return {
            med.lower().strip() for med in raw_output.split(",") if len(med.strip()) > 0
        }

    def get_central_index(med_begin: int, token_index_ls: list[tuple[int, int]]) -> int:
        def closest(token_ord: int) -> int:
            return abs(token_index_ls[token_ord][0] - med_begin)

        return min(range(len(token_index_ls)), default=-1, key=closest)

    def build_med_windows(
        section_begin: int, section_body: str, meds: set[str]
    ) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
        meds_regex = "|".join(map(re.escape, meds))
        normalized_section = section_body.lower()
        token_index_ls = [token.span() for token in re.finditer(r"\S+", section_body)]

        def match_to_window(med_match) -> tuple[tuple[int, int], tuple[int, int], str]:
            # DONE - adjust these to document level offsets by using
            # section offsets
            med_local_begin, med_local_end = med_match.span()
            med_central_index = get_central_index(med_local_begin, token_index_ls)
            # DONE - adjust these to document level offsets by using
            # section offsets
            med_cas_begin = med_local_begin + section_begin
            med_cas_end = med_local_end + section_begin
            window_local_begin = token_index_ls[
                max(0, med_central_index - WINDOW_RADIUS)
            ][0]
            window_local_end = token_index_ls[
                min(len(token_index_ls) - 1, med_central_index + WINDOW_RADIUS)
            ][1]
            window_cas_begin = window_local_begin + section_begin
            window_cas_end = window_local_end + section_begin
            opening = normalized_section[window_local_begin:med_local_begin]
            tagged_medication = normalized_section[med_local_begin:med_local_end]
            closing = normalized_section[med_local_end:window_local_end]
            window = (
                f"...{opening}<medication>{tagged_medication}</medication>{closing}..."
            )
            return (
                (med_cas_begin, med_cas_end),
                (window_cas_begin, window_cas_end),
                window,
            )

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
        section_offsets = literal_eval(row["section_offsets"])
        section_begin, sectio_end = section_offsets
        return build_med_windows(section_begin, str(row.section_body), meds)

    raw_frame["raw_windows"] = raw_frame.apply(row_to_window_list, axis=1)

    def non_empty(s: list[Any]) -> bool:
        return len(s) > 0

    # raw_frame[~raw_frame["raw_windows"].map(non_empty)][["serialized_output", "section_body"]].to_csv("/home/etg/Repos/CRICO/testing_escape/problem_children.tsv", sep="\t")
    raw_frame = raw_frame[raw_frame["raw_windows"].map(non_empty)]
    full_frame = raw_frame.explode("raw_windows")

    full_frame["medication_cas_offsets"] = full_frame["raw_windows"].map(itemgetter(0))

    full_frame["window_cas_offsets"] = full_frame["raw_windows"].map(itemgetter(1))

    full_frame["window_text"] = full_frame["raw_windows"].map(itemgetter(2))

    full_frame.drop("raw_windows", axis=1, inplace=True)
    full_frame.reset_index(drop=True)
    return full_frame


def agent_1_to_agent_2(input_tsv: str, output_dir: str) -> None:
    raw_frame = pd.read_csv(input_tsv, sep="\t")
    expanded_windows_frame = build_frame_with_med_windows(raw_frame)
    mkdir(output_dir)
    input_file_basename = basename_no_ext(input_tsv)
    out_path = os.path.join(output_dir, f"windowed_{input_file_basename}.tsv")
    expanded_windows_frame.to_csv(out_path, sep="\t", index=False)


def main() -> None:
    args = parser.parse_args()
    match args.input_mode, args.output_mode:
        case "agent_2_output", "anafora_xml":
            assert args.input_tsv is not None and args.output_dir is not None, (
                f"input tsv is {args.input_tsv} and output dir is {args.output_dir} both must be non-None"
            )
            agent_2_to_anafora(args.input_tsv, args.output_dir, args.get_differences)
        case "agent_1_output", "agent_2_output":
            assert args.input_tsv is not None and args.output_dir is not None, (
                f"input tsv is {args.input_tsv} and output dir is {args.output_dir} both must be non-None"
            )
            agent_1_to_agent_2(args.input_tsv, args.output_dir)
        case "anafora_xml", "json_lines":
            assert (
                args.paired_anafora_dir is not None and args.output_dir is not None
            ), (
                f"input tsv is {args.paired_anafora_dir} and output dir is {args.output_dir} both must be non-None"
            )
            anafora_to_json_lines(args.paired_anafora_dir, args.output_dir)
        case "agent_2_output", "json_lines":
            assert args.input_tsv is not None and args.output_dir is not None, (
                f"input tsv is {args.input_tsv} and output dir is {args.output_dir} both must be non-None"
            )
            agent_2_to_json_lines(args.input_tsv, args.output_dir)
        case _:
            logger.info(
                f"No transformations defined from {args.input_mode} to {args.output_mode} - skipping"
            )


if __name__ == "__main__":
    main()

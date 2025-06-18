import argparse
import os
import json
import pandas as pd
from lxml import etree
from lxml.etree import (
    _Element,
)  # Don't you love it when a language/library makes you feel

# like a perpetrator of antipatterns for using types?
import logging
from collections.abc import Iterable
from functools import partial


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--paired_anafora_dir",
    type=str,
    default=None,
)

parser.add_argument(
    "--input_tsv",
    type=str,
    default=None,
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
)


def __build_medication_dictionary_from_anafora(
    note_text: str,
    study_id: int,
    medication_annotation: _Element,
) -> dict[str, str | int | bool]:
    begin, end = [
        int(idx) for idx in medication_annotation.find("span").text.split(",")
    ]
    properties = medication_annotation.find("properties")
    return {
        "study_id": study_id,
        "medication": note_text[begin:end].strip().lower(),
        "has_at_least_one_instruction": any(
            p.text for p in properties.findall("instruction_")
        ),
        "has_at_least_one_condition": any(
            p.text for p in properties.findall("instruction_condition")
        ),
    }


def __build_medication_dictionary_from_tsv(
    row: pd.Series,
) -> dict[str, str | int | bool]:
    if len(row.medication) == 0:
        logger.warning(f"Empty medication found in {row.filename}")
    return {
        "study_id": int(row.filename.split("_")[-1]),
        "medication": row.medication.strip().lower(),
        "has_at_least_one_instruction": len(row.instruction) > 0,
        "has_at_least_one_condition": len(row.condition) > 0,
    }


# In [50]: import json
# In [51]: json.dumps({"_json_true": True, "_json_false_": False})
# Out[51]: '{"_json_true": true, "_json_false_": false}'
# It gets taken care of
def __file_to_dictionaries(
    study_id: int, xml_path: str, note_path: str
) -> Iterable[dict[str, str | int | bool]]:
    with open(xml_path, mode="rb") as xml_f:
        anafora_xml = etree.fromstring(xml_f.read())

    with open(note_path, mode="rt") as note_f:
        note_text = note_f.read()

    __local_med_dict = partial(
        __build_medication_dictionary_from_anafora,
        note_text,
        study_id,
    )
    for annotation in anafora_xml.find("annotations"):
        if (
            annotation.tag == "entity"
            and annotation.find("type").text == "Medications/Drugs"
        ):
            yield __local_med_dict(annotation)


def __get_med_json_line(medication_dictionary: dict[str, str | int | bool]) -> str:
    return f"{json.dumps(medication_dictionary)}\n"


def __dir_to_dictionaries(
    anafora_dir: str,
) -> Iterable[dict[str, str | int | bool]]:
    def get_study_id(subdir: str) -> int:
        return int(subdir.split("_")[-1])

    def get_xml_and_note_fns(abs_subdir: str) -> tuple[str, str]:
        relevant_files = (fn for fn in os.listdir(abs_subdir) if fn.startswith("Study"))
        xml_fn, note_fn = sorted(relevant_files, reverse=True)
        return xml_fn, note_fn

    for subdir in os.listdir(anafora_dir):
        if subdir.startswith("Study"):
            study_id = get_study_id(subdir)
            abs_subdir = os.path.join(anafora_dir, subdir)
            xml_fn, note_fn = get_xml_and_note_fns(abs_subdir)
            xml_path = os.path.join(abs_subdir, xml_fn)
            note_path = os.path.join(abs_subdir, note_fn)
            yield from __file_to_dictionaries(
                study_id=study_id, xml_path=xml_path, note_path=note_path
            )


def __anafora_process(
    paired_anafora_dir: str,
    output_dir: str,
) -> None:
    out_path = os.path.join(output_dir, f"{os.path.basename(paired_anafora_dir)}.jsonl")

    with open(out_path, mode="wt", encoding="utf-8") as f:
        for medication_dictionary in __dir_to_dictionaries(paired_anafora_dir):
            f.write(__get_med_json_line(medication_dictionary))


def __tsv_process(
    input_tsv: str,
    output_dir: str,
) -> None:
    df = pd.read_csv(input_tsv, sep="\t").fillna("")

    out_path = os.path.join(output_dir, f"{os.path.basename(input_tsv)}.jsonl")

    with open(out_path, mode="wt", encoding="utf-8") as f:
        for medication_dictionary in df.apply(
            __build_medication_dictionary_from_tsv, axis=1
        ):
            f.write(__get_med_json_line(medication_dictionary))


def main() -> None:
    args = parser.parse_args()
    if args.paired_anafora_dir is not None:
        __anafora_process(args.paired_anafora_dir, args.output_dir)
    elif args.input_tsv is not None:
        __tsv_process(args.input_tsv, args.output_dir)


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import os
import pathlib
import re
from ast import literal_eval
from functools import partial
from operator import itemgetter

import pandas as pd
from write_to_org import deserialize

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--excel_input",
    type=str,
)
parser.add_argument("--output_dir", type=str)
# parser.add_argument("--parse_type", choices=["json", "xml", "xml_first", "json_first"])


def parse_serialized_output(serialized_output: str) -> tuple[str, str]:
    json_raw_parse, xml_raw_parse = [
        text_body.strip()
        for text_body in re.split(
            r"XML\:|JSON\:",
            deserialize(literal_eval(serialized_output)[0]),
        )
        if len(text_body.strip()) > 0
    ]
    return json_raw_parse, xml_raw_parse


def parse_key_from_json(key: str, json_str) -> str:
    try:
        raw_dict = json.loads(json_str)
        return " , ".join(raw_dict.get(key, []))
    except Exception:
        return ""


def process(excel_input: str, output_dir: str) -> None:
    df = (
        pd.read_excel(excel_input)
        if excel_input.lower().endswith("xlsx")
        else pd.read_csv(excel_input, sep="\t")
    )
    input_basename = pathlib.Path(excel_input).stem.strip()
    reference_path = os.path.join(
        output_dir, f"REFERENCE_survey_ready_{input_basename}.xlsx"
    )
    summarized_path = os.path.join(output_dir, f"survey_ready_{input_basename}.xlsx")

    df["combined"] = df["serialized_output"].map(parse_serialized_output)
    df["JSON"] = df["combined"].map(itemgetter(0))
    df["XML"] = df["combined"].map(itemgetter(1))
    attrs = [
        "medication",
        "dosage",
        "frequency",
        "condition",
        "instruction",
    ]
    for attr in attrs:
        df[attr] = df["JSON"].map(partial(parse_key_from_json, attr))

    def clean_section_id(section_str: str) -> str:
        return " ".join(section_str.split("_")[1:]).title()

    df["section_identifier"] = df["section_identifier"].map(clean_section_id)
    reference_df = df[
        ["filename", "section_identifier", *attrs, "window_text", "JSON", "XML"]
    ]
    summarized_df = df[
        ["filename", "section_identifier", *attrs, "window_text", "JSON", "XML"]
    ]
    reference_df.to_excel(reference_path, index=False)
    summarized_df.to_excel(summarized_path, index=False)


def main() -> None:
    args = parser.parse_args()
    process(args.excel_input, args.output_dir)


if __name__ == "__main__":
    main()

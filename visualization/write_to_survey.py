import argparse
import json
import logging
import os
import pathlib
import re
from ast import literal_eval
from itertools import chain
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
parser.add_argument("--filter_hallucinations", action="store_true")
# parser.add_argument("--parse_type", choices=["json", "xml", "xml_first", "json_first"])


def serialize_whitespace(sample: str | None) -> str:
    if sample is None:
        return "None"
    return (
        sample.replace("\n", "<cn>")
        .replace("\t", "<ct>")
        .replace("\f", "<cf>")
        .replace("\r", "<cr>")
    )


def field_is_hallucinatory(
    row: pd.Series, ground_truth_column: str, field: str
) -> bool:
    def normalize_str(sample: str) -> str:
        return sample.strip().lower()

    return normalize_str(row[field]) in normalize_str(row[ground_truth_column])


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


def parse_key_from_json(key: str, json_str) -> str:
    try:
        raw_dict = json.loads(json_str)
        return " , ".join(raw_dict.get(key, []))
    except Exception:
        return ""


def get_medication(window_text: str) -> str:
    matches = re.findall(r"<medication>(.+)</medication>", window_text)
    if len(matches) == 0:
        return ""
    return matches[0]


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


def process(excel_input: str, output_dir: str, filter_hallucinations: bool) -> None:
    df = (
        pd.read_excel(excel_input)
        if excel_input.lower().endswith("xlsx")
        else pd.read_csv(excel_input, sep="\t")
    )
    input_basename = pathlib.Path(excel_input).stem.strip()
    # reference_path = os.path.join(
    #     output_dir, f"REFERENCE_survey_ready_{input_basename}.xlsx"
    # )
    # summarized_path = os.path.join(output_dir, f"survey_ready_{input_basename}.xlsx")

    reference_path = os.path.join(
        output_dir,
        f"survey_ready_{input_basename}_hallucinatory_attributes_filtered.tsv"
        if filter_hallucinations
        else f"survey_ready_{input_basename}.tsv",
    )

    def get_id_number(fn: str) -> int:
        return int(fn.split("_")[-1])

    df["medication"] = df["window_text"].map(get_medication)
    df = df.loc[df["medication"] != ""]
    df["combined"] = df["serialized_output"].map(parse_serialized_output)
    df["JSON"] = df["combined"].map(itemgetter(0))
    df["XML"] = df["combined"].map(itemgetter(1))
    df.drop(columns=["combined", "serialized_output"], inplace=True)
    df["JSON"] = df.apply(select_json, axis=1)
    df["XML"] = df.apply(select_xml, axis=1)
    attrs = [
        "medication",
        "dosage",
        "frequency",
        "condition",
        "instruction",
    ]
    for attr in attrs:
        if attr != "medication":
            df[attr] = df["JSON"].map(partial(parse_key_from_json, attr))
            if filter_hallucinations:
                df[f"{attr}_hallucinatory"] = df.apply(
                    partial(
                        field_is_hallucinatory,
                        ground_truth_column="window_text",
                        field=attr,
                    ),
                    axis=1,
                )
                # df.loc[df[f"{attr}_hallucinatory"], attr] = ""
    df = df.loc[(df["instruction"] != "") | (df["condition"] != "")]

    def clean_section_id(section_str: str) -> str:
        return " ".join(section_str.split("_")[1:]).title()

    non_med_attrs = set(attrs) - {"medication"}
    attr_hallucinations = {f"{attr}_hallucinatory" for attr in non_med_attrs}
    df["section_identifier"] = df["section_identifier"].map(clean_section_id)
    # df.drop(columns=[f"{attr}_hallucinatory" for attr in set(attrs) - {"medication"}], inplace=True)
    for column_name in df.columns:
        if not column_name.endswith("hallucinatory"):
            df[column_name] = df[column_name].map(serialize_whitespace)
    reference_df = df[
        ["filename", "section_identifier", *attrs, "window_text", "JSON", "XML"]
    ]
    reference_df["int_study_id"] = reference_df["filename"].map(get_id_number)
    reference_df = reference_df.sort_values(by="int_study_id")
    # summarized_df = df[
    #     ["filename", "section_identifier", *attrs, "window_text", "JSON", "XML"]
    # ]
    reference_df = df[
        [
            "filename",
            "section_identifier",
            "medication",
            *chain.from_iterable(
                zip(sorted(non_med_attrs), sorted(attr_hallucinations))
            ),
            "window_text",
            "JSON",
            "XML",
        ]
    ]
    # reference_df.to_excel(reference_path, index=False)
    # summarized_df.to_excel(summarized_path, index=False)
    reference_df.to_csv(reference_path, sep="\t", index=False)


def main() -> None:
    args = parser.parse_args()
    process(args.excel_input, args.output_dir, args.filter_hallucinations)


if __name__ == "__main__":
    main()

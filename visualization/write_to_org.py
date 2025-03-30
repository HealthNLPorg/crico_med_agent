import os
from ast import literal_eval
import json
import textwrap
import pathlib
import pandas as pd
import argparse
import logging
from typing import cast, Iterable

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--excel_input",
    type=str,
)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--parse_type", choices=["json", "xml"])
parser.add_argument("--input_sample_column", type=str)

def deserialize(s: str) -> str:
    return (
        s.replace("<cn>", "\n")
        .replace("<cr>", "\r")
        .replace("<ct>", "\t")
        .replace("<cf>", "\f")
    )


class ParsedSection:
    def __init__(
        self,
        input_sample: str,
        section_identifier: str,
        parse_matter: str,
        parse_type: str,
    ) -> None:
        self.input_sample = input_sample
        self.section_identifier = section_identifier
        self.parse_matter = parse_matter
        self.parse_type = parse_type
        # json_output if isinstance(json_output, list) else []

    def to_org_node(self, base_depth: int = 2, width: int = 120) -> str:
        def mention_to_str(mention_dict: dict[str, str | list[str]]) -> str:
            def list_to_str(attr_list: Iterable[str]) -> str:
                return (
                    "\n".join(
                        f"{idx + 1}. {attr}" for idx, attr in enumerate(attr_list)
                    )
                    + "\n"
                )

            return (
                f"Medication: {mention_dict.get('medication', 'ERROR_MENTION_WITH_NO_MEDICATION')}\n"
                f"Instructions:\n{list_to_str(mention_dict.get('instructions', []))}"
                f"Conditions:\n{list_to_str(mention_dict.get('conditions', []))}"
            )

        match self.parse_type:
            case "json":
                json_output = (
                    self.parse_matter if isinstance(self.parse_matter, list) else []
                )
                parsed = (
                    "\n".join(
                        (mention_to_str(mention_dict) for mention_dict in json_output)
                    )
                    if len(json_output) > 0
                    else "None"
                )
            case "xml":
                contained = literal_eval(self.parse_matter)[0]
                parsed = (
                    contained
                    if contained is None
                    else textwrap.fill(deserialize(contained).strip(), width=width)
                )
        return (
            f"{base_depth * '*'} {self.section_identifier}:\n"
            f"{(base_depth + 1) * '*'} Input Sample:\n"
            f"{textwrap.fill(deserialize(self.input_sample), width=width)}\n"
            f"{(base_depth + 1) * '*'} Model Output:\n"
            f"{parsed}\n"
            f"{(base_depth + 1) * '*'} TODO Error Analysis:\n\n\n"
        )


def json_parse(row: pd.Series) -> ParsedSection:
    try:
        raw_json = json.loads(row.json_output)
        parsed_json = raw_json if isinstance(raw_json, list) else []
    except Exception:
        try:
            raw_json = literal_eval(row.json_output)
            parsed_json = raw_json if isinstance(raw_json, list) else []
        except Exception:
            parsed_json = []
    finally:
        return ParsedSection(
            row.input_sample, row.section_identifier, parsed_json, "json"
        )


def xml_parse(row: pd.Series, input_sample_column: str) -> ParsedSection:
    return ParsedSection(
        row[input_sample_column], row.section_identifier, row.serialized_output, "xml"
    )


def process(excel_input: str, output_dir: str, parse_type: str, input_sample_column: str) -> None:
    df = (
        pd.read_excel(excel_input)
        if excel_input.lower().endswith("xlsx")
        else pd.read_csv(excel_input, sep="\t")
    )
    input_basename = pathlib.Path(excel_input).stem.strip()
    output_path = os.path.join(output_dir, f"{input_basename}.org")

    match parse_type:
        case "json":

            def get_parsed_section(row):
                return json_parse(row, input_sample_column)
        case "xml":

            def get_parsed_section(row):
                return xml_parse(row, input_sample_column)

    with open(output_path, mode="w") as f:
        for fn, fn_frame in df.groupby(["filename"]):
            (base_fn,) = cast(tuple[str,], fn)
            f.write(f"* TODO {base_fn}\n")
            for parsed_section in fn_frame.apply(get_parsed_section, axis=1):
                f.write(parsed_section.to_org_node())


def main() -> None:
    args = parser.parse_args()
    process(args.excel_input, args.output_dir, args.parse_type, args.input_sample_column)


if __name__ == "__main__":
    main()

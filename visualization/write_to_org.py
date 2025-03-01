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


class ParsedSection:
    def __init__(
        self, section_body: str, section_identifier: str, json_output: str
    ) -> None:
        self.section_body = section_body
        self.section_identifier = section_identifier
        self.json_output = json_output if isinstance(json_output, list) else []

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

        return (
            f"{base_depth * '*'} {self.section_identifier}:\n"
            f"{(base_depth + 1) * '*'} Section Body:\n"
            f"{textwrap.fill(self.section_body, width=width)}\n"
            f"{(base_depth + 1) * '*'} Model Output:\n"
            f"{'\n'.join((mention_to_str(mention_dict) for mention_dict in self.json_output)) if len(self.json_output) > 0 else 'None'}\n"
            f"{(base_depth + 1) * '*'} TODO Error Analysis:\n\n\n"
        )


def process(excel_input: str, output_dir: str) -> None:
    df = pd.read_excel(excel_input)
    input_basename = pathlib.Path(excel_input).stem.strip()
    output_path = os.path.join(output_dir, f"{input_basename}.org")

    def get_parsed_section(row: pd.Series) -> ParsedSection:
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
            return ParsedSection(row.section_body, row.section_identifier, parsed_json)

    with open(output_path, mode="w") as f:
        for fn, fn_frame in df.groupby(["filename"]):
            (base_fn,) = cast(tuple[str,], fn)
            f.write(f"* TODO {base_fn}\n")
            for parsed_section in fn_frame.apply(get_parsed_section, axis=1):
                f.write(parsed_section.to_org_node())


def main() -> None:
    args = parser.parse_args()
    process(args.excel_input, args.output_dir)


if __name__ == "__main__":
    main()

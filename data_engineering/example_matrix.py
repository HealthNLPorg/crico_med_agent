import re
import argparse
import json
import pathlib

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_file",
    type=str,
)
parser.add_argument("--output_dir", type=str)


class JSONExample:
    def __init__(self, finalized_body: dict[str, list[str]] | None) -> None:
        self.finalized_body = finalized_body

    def __str__(self) -> str:
        if self.finalized_body is None:
            return "None"
        return json.dumps(self.finalized_body)


def parse_input_output(examples_file_path: str) -> list[tuple[str, str]]:
    def parse_example(raw_example: str) -> tuple[str, str]:
        result = tuple(
            elem.strip()
            for elem in re.split("input:|output:", raw_example)
            if len(elem.strip()) > 0
        )
        assert len(result) == 2
        return result

    with open(examples_file_path, mode="rt", encoding="utf-8") as ef:
        no_comments_str = "".join(
            line for line in ef.readlines() if not line.strip().startswith("#")
        )
    return [
        parse_example(example.strip())
        for example in no_comments_str.split("\n\n")
        if len(example.split()) > 0
    ]


def parse_output_dict(output_str: str) -> dict[str, str]:
    xml_example, json_example = [
        text_body.strip()
        for text_body in re.split(r"XML\:|JSON\:", output_str)
        if len(text_body) > 0
    ]
    return {"XML": xml_example, "JSON": json_example}


def normalize_json_dict(serialized_json: str) -> dict[str, list[str]] | None:
    raw_dictionary = json.loads(serialized_json)
    medication_value = raw_dictionary.get("medication", [])
    instruction_value = raw_dictionary.get("instruction", [])
    condition_value = raw_dictionary.get("instructionCondition", [])
    if len(medication_value) == 0 or (
        len(instruction_value) == 0 and len(condition_value) == 0
    ):
        return None
    dosage_value = raw_dictionary.get("dosage", [])
    frequency_value = raw_dictionary.get("frequency", [])
    return {
        "medication": medication_value,
        "instruction": instruction_value,
        "condition": condition_value,
        "dosage": dosage_value,
        "frequency": frequency_value,
    }


def write_outputs(
    inputs: list[str],
    parsed_outputs: list[dict[str, JSONExample | str]],
    output_dir: str,
    input_fn: str,
) -> None:
    pass


def process(input_file: str, output_dir: str) -> None:
    raw_example_list = parse_input_output(input_file)
    inputs = [example[0] for example in raw_example_list]
    raw_outputs = [example[1] for example in raw_example_list]

    def full_parse(output: str) -> dict[str, str | JSONExample]:
        raw_dict = parse_output_dict(output)
        return {
            "XML": raw_dict.get("XML"),
            "JSON": JSONExample(normalize_json_dict(raw_dict.get("JSON"))),
        }

    fully_parsed_outputs = [full_parse(output) for output in raw_outputs]
    input_fn = pathlib.Path(input_file).stem
    write_outputs(inputs, fully_parsed_outputs, output_dir, input_fn)


def main() -> None:
    args = parser.parse_args()
    process(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()

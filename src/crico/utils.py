import pathlib
import re
from ast import literal_eval


# retain newline information via special markers
# while removing them for storage
# ( so you can load them later via pandas without parsing errors )
# ignoring vertical tabs (\v) for now
# unless we run into them
def serialize_whitespace(sample: str | None) -> str:
    if sample is None:
        return "None"
    return (
        sample.replace("\n", "<cn>")
        .replace("\t", "<ct>")
        .replace("\f", "<cf>")
        .replace("\r", "<cr>")
    )


def deserialize_whitespace(sample: str | None) -> str:
    if sample is None:
        return "None"
    return (
        sample.replace("<cn>", "\n")
        .replace("<ct>", "\t")
        .replace("<cf>", "\f")
        .replace("<cr>", "\r")
    )


def parse_serialized_output(serialized_output: str) -> tuple[list[str], list[str]]:
    model_output = deserialize_whitespace(literal_eval(serialized_output)[0])
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


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def mkdir(dir_name: str) -> None:
    _dir_name = pathlib.Path(dir_name)
    _dir_name.mkdir(parents=True, exist_ok=True)

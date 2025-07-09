import pathlib


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


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def mkdir(dir_name: str) -> None:
    _dir_name = pathlib.Path(dir_name)
    _dir_name.mkdir(parents=True, exist_ok=True)

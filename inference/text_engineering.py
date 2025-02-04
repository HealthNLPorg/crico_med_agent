# retain newline information via special markers
# while removing them for storage
# ( so you can load them later via pandas without parsing errors )
# ignoring vertical tabs (\v) for now
# unless we run into them
def serialize_whitespace(sample: str) -> str:
    return (
        sample.replace("\n", "<cn>")
        .replace("\t", "<ct>")
        .replace("\f", "<cf>")
        .replace("\r", "<cr>")
    )


def deserialize_whitespace(sample: str) -> str:
    return (
        sample.replace("<cn>", "\n")
        .replace("<ct>", "\t")
        .replace("<cf>", "\f")
        .replace("<cr>", "\r")
    )

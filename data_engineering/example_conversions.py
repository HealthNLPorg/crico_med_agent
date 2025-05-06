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

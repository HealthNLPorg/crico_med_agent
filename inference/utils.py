import pathlib


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def mkdir(dir_name: str) -> None:
    _dir_name = pathlib.Path(dir_name)
    _dir_name.mkdir(parents=True, exist_ok=True)

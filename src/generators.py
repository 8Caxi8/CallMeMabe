import json
from typing import Generator


def get_prompt(path: str) -> Generator[dict, None, None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        for prompt in data:
            yield prompt["prompt"]


def get_functions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        functions = json.load(f)

    return functions

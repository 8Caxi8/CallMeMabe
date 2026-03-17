import sys
import json


def parser() -> tuple[str, str, str]:
    args = sys.argv[1:]

    functions = "data/input/functions_definition.json"
    input_file = "data/input/function_calling_tests.json"
    output_file = "data/output/function_calls.json"

    for i, arg in enumerate(args):
        if arg == "--functions_definition":
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {arg}")
            functions = args[i + 1]
        elif arg == "--input":
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {arg}")
            input_file = args[i + 1]
        elif arg == "--output":
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {arg}")
            output_file = args[i + 1]

    return functions, input_file, output_file


def validate_json(*paths: str) -> None:
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                json.load(f)
        except FileNotFoundError:
            raise ValueError(f"File not found: {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in: {path}")

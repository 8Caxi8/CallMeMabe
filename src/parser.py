from typing import Any
import os
import sys
import json


def parser() -> tuple[list[dict[Any, Any]], list[dict[Any, Any]], str, str]:
    args = sys.argv[1:]
    i = 0

    functions_path = "data/input/functions_definition.json"
    input_file_path = "data/input/function_calling_tests.json"
    output_file_path = "data/output/function_calling_results.json"
    llm = "qwen3"

    while i < len(args):
        if args[i] == "--functions_definition":
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {args[i]}")
            i += 1
            functions_path = args[i]
        elif args[i] == "--input":
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {args[i]}")
            i += 1
            input_file_path = args[i]
        elif args[i] == "--output":
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {args[i]}")
            i += 1
            output_file_path = args[i]
        elif args[i] == "--llm":
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for {args[i]}")
            i += 1
            llm = args[i]
        i += 1

    functions = load_json(functions_path)
    input_file = load_json(input_file_path)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    return functions, input_file, output_file_path, llm


def load_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            file = json.load(f)

            if not file:
                raise ValueError(f"File in '{path}' is empty")
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in: {path}")

    return file


def output_json(output: list[dict[str, Any]], path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise ValueError(f"Could not write output file: {path}\n{e}")

from typing import Any


def print_header(prompt: str) -> None:
    print()
    print("="*120)
    print("*"*5, end="")
    print(f"Calculating for prompt: {prompt}", end="")
    print("*" * 5)


def print_outcome(result: dict[str, Any]) -> None:
    print()
    print(f"  - 'prompt': '{result['prompt']}'")
    print(f"  - 'name': '{result['name']}")
    print("  - 'parameters':")
    for key, value in result["parameters"].items():
        print(f"    - '{key}': {repr(value)}")

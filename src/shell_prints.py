from typing import Any
from time import sleep

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
PURPLE = "\033[35m"
RESET = "\033[0m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
CURSOR_UP = "\033[A"
CLEAR_LINE = "\033[2K"


def print_failed_outcome(i: int, prompt: str, e: str) -> None:
    print(f"{RED}{BOLD}[✗] FAILED !{RESET}")
    print(f"{RED}{e}{RESET}")


def print_fallback(**kwargs) -> None:
    for key, value in kwargs.items():
        print(f"\r{RED}[WARNING]: {key} triggered (value={value})", end="",
              flush=True)


def print_header(i: int, prompt: str) -> int:
    print(f"{BLUE}{'='*120}")
    print(f"[#{i}] Prompt: '{prompt}'{RESET}")
    return start_progress(prompt)


def print_success_outcome(result: dict[str, Any]) -> None:
    print(f"{GREEN}{BOLD}[✓] SUCCESS !{RESET}")
    print(f"{GREEN}  - 'prompt': '{result['prompt']}'")
    print(f"  - 'name': '{result['name']}'")
    print("  - 'parameters':")
    for key, value in result["parameters"].items():
        print(f"    - '{key}': {repr(value)}")
    print(RESET, end="")
    sleep(0.5)


def print_llm_initializer(prompt_size: int, func_no: int) -> None:
    print(f"{ITALIC}{PURPLE}Initializing llm 'Qwen/Qwen3-0.6B' ...{RESET}\n")


def print_progress(msg: str, end_l: str) -> int:
    print(f"{YELLOW}{msg}{RESET}", end=end_l, flush=True)

    return 1


def start_progress(prompt: str) -> int:
    print_progress("[?] IN PROGRESS ...", "\n")
    print_progress(f"  - 'prompt': {prompt}", "\n")
    print_progress("  - 'name': 'fn_", "")

    return 3


def clear_lines(n: int) -> int:
    for _ in range(n - 1):
        print(f"\r{CLEAR_LINE}{CURSOR_UP}", end="", flush=True)
    print(f"\r{CLEAR_LINE}", end="", flush=True)

    return 0

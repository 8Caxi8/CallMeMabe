from typing import Any
from time import sleep

VERBOSE = False
LINES = 0
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
    """Print a failure message for a prompt that could not be processed.

    Args:
        i: The prompt index.
        prompt: The original prompt string.
        e: The error message to display.
    """

    print(f"{RED}{BOLD}[✗] FAILED !{RESET}")
    print(f"{RED}{e}{RESET}")


def print_fallback(**kwargs: Any) -> None:
    """Print an inline error when a generation limit is triggered.

    Args:
        **kwargs: Either loop_counter or tokens_counter with their limit value.
    """

    for key, value in kwargs.items():
        print(f"{CLEAR_LINE}{RED}[ERROR]: {key} triggered (value={value})",
              end="", flush=True)


def print_header(i: int, prompt: str) -> None:
    """Print the prompt header and start the in-progress display.

    Args:
        i: The prompt index.
        prompt: The original prompt string.
    """

    print(f"{BLUE}{'='*120}")
    print(f"[#{i}] Prompt: '{prompt}'{RESET}")
    start_progress(prompt)


def print_success_outcome(result: dict[str, Any]) -> None:
    """Print a success message with the full function call result.

    Args:
        result: Dict containing 'prompt', 'name', and 'parameters'.
    """

    print(f"{GREEN}{BOLD}[✓] SUCCESS !{RESET}")
    print(f"{GREEN}  - 'prompt': '{result['prompt']}'")
    print(f"  - 'name': '{result['name']}'")
    print("  - 'parameters':")
    for key, value in result["parameters"].items():
        print(f"    - '{key}': {repr(value)}")
    print(RESET, end="")
    sleep(0.5)


def print_llm_initializer(name: str) -> None:
    """Print a message when the LLM is being loaded.

    Args:
        name: The model name being initialized.
    """

    print(f"{ITALIC}{PURPLE}Initializing llm '{name}' ...{RESET}\n")


def print_new_progress(msg: str, end_l: str) -> None:
    """Print a progress message and increment the line counter.

    Args:
        msg: The message to print.
        end_l: The line ending character.
    """

    print(f"{YELLOW}{msg}{RESET}", end=end_l, flush=True)
    global LINES
    LINES += 1


def print_progress(msg: str, end_l: str) -> None:
    """Print a progress message without incrementing the line counter.

    Args:
        msg: The message to print.
        end_l: The line ending character.
    """

    print(f"{YELLOW}{msg}{RESET}", end=end_l, flush=True)


def print_recover(msg: str, end_l: str) -> None:
    """Print a warning when the LLM output is being recovered.

    Args:
        msg: The warning message to display.
        end_l: The line ending character.
    """

    print(f"\n{RED}[WARNING]: {msg}{RESET}", end=end_l, flush=True)
    sleep(2)
    global LINES
    LINES += 1


def start_progress(prompt: str) -> None:
    """Print the initial in-progress lines for a prompt.

    Args:
        prompt: The original prompt string.
    """

    print_progress("[?] IN PROGRESS ...", "\n")
    print_progress(f"  - 'prompt': {prompt}", "\n")
    print_progress("  - 'name': 'fn_", "")

    global LINES
    LINES += 3


def clear_lines() -> None:
    """Clear the in-progress lines from the terminal.

    In verbose mode prints a separator instead of clearing,
    since the constrained decoding output should remain visible.
    """

    global LINES
    if VERBOSE:
        print(f"\n{BLUE}{'='*120}{RESET}")
        LINES = 0
        return
    for _ in range(LINES - 1):
        print(f"\r{CLEAR_LINE}{CURSOR_UP}", end="", flush=True)
    print(f"\r{CLEAR_LINE}", end="", flush=True)
    LINES = 0


def print_output(path: str) -> None:
    """Print a message when the output file is being created.

    Args:
        path: The output file path.
    """

    print(f"\n{ITALIC}{PURPLE}Creating '{path}' ...{RESET}\n")


def set_verbose(verbose: bool) -> None:
    """Set the global verbose mode for constrained decoding output.

    When enabled, print_constrained_step will print each token
    selection with its logit score and masking info.

    Args:
        verbose: True to enable verbose mode, False to disable.
    """

    global VERBOSE
    VERBOSE = verbose


def print_constrained_step(token_str: str, logit: float,
                           masked: int, candidates: int) -> None:
    """Print one constrained decoding step in verbose mode.

    Shows the selected token, its logit score, how many tokens
    were masked as invalid, and how many valid candidates remained.
    No-ops when verbose mode is disabled.

    Args:
        token_str: The token string that was selected.
        logit: The logit score of the selected token.
        masked: Number of tokens masked to -inf at this step.
        candidates: Number of valid tokens remaining after masking.
    """

    if not VERBOSE:
        return
    print(f"\n{CYAN}[CONSTRAINED] selected: '{token_str}' "
          f"(logit: {logit:.2f}) | "
          f"masked: {masked} | "
          f"valid candidates: {candidates}{RESET} --> ", end="", flush=True)
    global LINES
    LINES += 2
    sleep(0.3)

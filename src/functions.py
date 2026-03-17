from math import sqrt
import re


def fn_add_numbers(a: float, b: float) -> float:
    return a + b


def fn_greet(name: str) -> str:
    return f"Greetings {name}!"


def fn_reverse_string(s: str) -> str:
    return s[::-1]


def fn_get_square_root(a: float) -> float:
    if a < 0:
        raise ValueError("Cannot compute square root "
                         f"for negative numbers: {a}")
    return sqrt(a)


def fn_substitute_string_with_regex(source_string: str, regex: str,
                                    replacement: str) -> str:
    return re.sub(regex, replacement, source_string)


FUNCTIONS = {
    "fn_add_numbers": fn_add_numbers,
    "fn_greet": fn_greet,
    "fn_reverse_string": fn_reverse_string,
    "fn_get_square_root": fn_get_square_root,
    "fn_substitute_string_with_regex": fn_substitute_string_with_regex,
}

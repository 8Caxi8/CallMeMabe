from .shell_prints import (print_progress, print_fallback,
                           print_constrained_step)
from .validation_models import ParameterType, FunctionsDefinition
from .llm import BaseLLM
from typing import Callable
import json


MAX_TOKENS = 100
MAX_LOOP_ITER = 50


def get_function_name(valid_func: list[FunctionsDefinition],
                      prompt: str,
                      model: BaseLLM) -> str:
    """Select the correct function name for a prompt using constrained
    decoding.

    Builds the function name token by token, only accepting tokens that
    extend a valid function name prefix. Falls back to the first available
    function if the loop or token limits are exceeded.

    Args:
        valid_func: List of validated function definitions to choose from.
        prompt: The natural language prompt to match against.
        model: The LLM wrapper used for token generation.

    Returns:
        The selected function name, e.g. 'fn_add_numbers'.
    """

    functions = [f.name for f in valid_func]
    descriptions = "\n".join(
        f"- {function.name.replace('fn_', '')}: {function.description}"
        for function in valid_func
    )
    tokens_counter: int = 0
    loop_counter: int = 0

    func_name = ""
    while "fn_" + func_name not in functions:
        if loop_counter > MAX_LOOP_ITER:
            print_fallback(loop_counter=MAX_LOOP_ITER)
            return functions[0]

        starting_string = (
            "You have to select the correct function by what is "
            "given in the prompt.\n"
            "Select a function name by keywords in the prompt "
            "EXACTLY as they are written.\n"
            f"{descriptions}\n"
            f"Prompt: {prompt}\n"
            f"Function to use: {func_name}"
        )

        input_ids = model.encode(starting_string)
        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return functions[0]
            token_id = logits.index(max(logits))

            token_str = model.get_cached_token(token_id,
                                               model.clean_function_name)
            masked_count = sum(1 for log in logits if log == float("-inf"))
            print_constrained_step(
                token_str,
                logits[token_id],
                masked_count,
                len(logits) - masked_count
            )

            candidate = "fn_" + func_name + token_str
            if any(f.startswith(candidate) for f in functions):
                func_name += token_str
                print_progress(token_str, "")
                break

            logits[token_id] = float("-inf")
            tokens_counter += 1

        loop_counter += 1

    return "fn_" + func_name


def get_delimited_parameter(model: BaseLLM,
                            starting_string: str,
                            start_delimiter: str,
                            end_delimiter: str) -> list[str]:
    """Extract an array or object parameter using delimiter-based generation.

    Generates tokens letter by letter until the end delimiter is reached.
    Falls back to an empty array or object if the loop limit is exceeded.

    Args:
        model: The LLM wrapper used for token generation.
        starting_string: The prompt context string.
        start_delimiter: Opening delimiter, either '[' or '{"'.
        end_delimiter: Closing delimiter, either ']' or '}'.

    Returns:
        List of characters forming the raw parameter value including
        delimiters.
    """

    parameter: list[str] = [start_delimiter]
    end_tokens = {end_delimiter}
    loop_counter: int = 0

    while True:
        if loop_counter > MAX_LOOP_ITER:
            print_fallback(loop_counter=MAX_LOOP_ITER)
            parameter.append(end_delimiter)
            return fallback_delimited(parameter,
                                      start_delimiter,
                                      end_delimiter)

        input_ids = model.encode(
            starting_string + "".join(
                parameter))
        logits = model.get_logits_from_input_ids(input_ids)

        token_id = logits.index(max(logits))
        token_str = model.get_cached_token(token_id,
                                           model.clean_str_tokens)
        masked_count = sum(1 for log in logits if log == float("-inf"))
        print_constrained_step(
            token_str,
            logits[token_id],
            masked_count,
            len(logits) - masked_count
        )

        for letter in token_str:
            parameter.append(letter)
            print_progress(letter, "")

            if letter in end_tokens:
                break

        if parameter and parameter[-1] in end_tokens:
            break

        loop_counter += 1

    return parameter


def get_number_parameter(model: BaseLLM,
                         starting_string: str) -> list[str]:
    """Extract a float parameter using constrained numeric decoding.

    Only accepts tokens that form a valid float candidate, allowing
    intermediate states like '-', '.', and '-.'. Stops at closing '"'.

    Args:
        model: The LLM wrapper used for token generation.
        starting_string: The prompt context string.

    Returns:
        List of characters forming the raw numeric value, excluding quotes.
    """

    parameter: list[str] = ["\""]
    end_tokens = {"\""}
    loop_counter: int = 0
    tokens_counter: int = 0

    while True:
        if loop_counter > MAX_LOOP_ITER:
            print_fallback(loop_counter=MAX_LOOP_ITER)
            return fallback_number(parameter, float)

        input_ids = model.encode(
            starting_string + "".join(
                parameter))

        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return fallback_number(parameter, float)

            token_id = logits.index(max(logits))
            token_str = model.get_cached_token(token_id,
                                               model.clean_number_tokens)
            masked_count = sum(1 for log in logits if log == float("-inf"))
            print_constrained_step(
                token_str,
                logits[token_id],
                masked_count,
                len(logits) - masked_count
            )
            candidate = "".join(parameter[1:] + [token_str])
            try:
                float(candidate)

            except ValueError:
                if candidate in {".", "-", "-."} or token_str in end_tokens:
                    pass
                else:
                    logits[token_id] = float("-inf")
                    tokens_counter += 1
                    continue

            for letter in token_str:
                parameter.append(letter)
                print_progress(letter, "")

                if letter in end_tokens:
                    break

            break

        if parameter and parameter[-1] in end_tokens:
            break

        loop_counter += 1

    return parameter[1:-1]


def get_int_parameter(model: BaseLLM,
                      starting_string: str) -> list[str]:
    """Extract an integer parameter using constrained numeric decoding.

    Only accepts tokens that form a valid integer candidate, allowing
    '-' as an intermediate state. Stops at closing '"'.

    Args:
        model: The LLM wrapper used for token generation.
        starting_string: The prompt context string.

    Returns:
        List of characters forming the raw integer value, excluding quotes.
    """

    parameter: list[str] = ["\""]
    end_tokens = {"\""}
    loop_counter: int = 0
    tokens_counter: int = 0

    while True:
        if loop_counter > MAX_LOOP_ITER:
            print_fallback(loop_counter=MAX_LOOP_ITER)
            return fallback_number(parameter, int)

        input_ids = model.encode(
            starting_string + "".join(
                parameter))
        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return fallback_number(parameter, int)

            token_id = logits.index(max(logits))

            token_str = model.get_cached_token(token_id,
                                               model.clean_number_tokens)
            masked_count = sum(1 for log in logits if log == float("-inf"))
            print_constrained_step(
                token_str,
                logits[token_id],
                masked_count,
                len(logits) - masked_count
            )

            candidate = "".join(parameter[1:] + [token_str])
            try:
                int(candidate)

            except ValueError:
                if token_str in end_tokens or candidate in {"-"}:
                    pass
                else:
                    logits[token_id] = float("-inf")
                    tokens_counter += 1
                    continue

            for letter in token_str:
                parameter.append(letter)
                print_progress(letter, "")
                if letter in end_tokens:
                    break

            break

        if parameter and parameter[-1] in end_tokens:
            break

        loop_counter += 1

    return parameter[1:-1]


def get_bool_parameter(model: BaseLLM,
                       starting_string: str) -> list[str]:
    """Extract a boolean parameter constrained to 'true' or 'false'.

    Masks all tokens that are not 'true' or 'false' until one is selected.
    Falls back to 'true' if the token limit is exceeded.

    Args:
        model: The LLM wrapper used for token generation.
        starting_string: The prompt context string.

    Returns:
        Single-element list containing either 'true' or 'false'.
    """

    parameter: list[str] = []
    tokens_counter: int = 0

    input_ids = model.encode(
        starting_string + "".join(
            parameter))
    logits = model.get_logits_from_input_ids(input_ids)

    while True:
        if tokens_counter > MAX_TOKENS:
            print_fallback(tokens_counter=MAX_TOKENS)
            return ["true"]

        token_id = logits.index(max(logits))
        token_str = model.get_cached_token(token_id,
                                           model.clean_number_tokens)
        masked_count = sum(1 for log in logits if log == float("-inf"))
        print_constrained_step(
            token_str,
            logits[token_id],
            masked_count,
            len(logits) - masked_count
        )

        if token_str.lower() not in {"true", "false"}:
            logits[token_id] = float("-inf")
            tokens_counter += 1
            continue

        parameter.append(token_str)
        print_progress(token_str, "")
        break

    return parameter


def get_string_parameter(model: BaseLLM,
                         starting_string: str) -> list[str]:
    """Extract a string parameter, stopping at an unescaped closing quote.

    Args:
        model: The LLM wrapper used for token generation.
        starting_string: The prompt context string.

    Returns:
        List of characters forming the raw string value, excluding quotes.
    """

    parameter: list[str] = ["\""]
    end_tokens = {"\""}
    loop_counter: int = 0

    while True:
        if loop_counter > MAX_LOOP_ITER:
            print_fallback(loop_counter=MAX_LOOP_ITER)
            return parameter[1:] if len(parameter) > 1 else [""]

        input_ids = model.encode(
            starting_string + "".join(
                parameter))
        logits = model.get_logits_from_input_ids(input_ids)

        token_id = logits.index(max(logits))
        token_str = model.get_cached_token(token_id,
                                           model.clean_str_tokens)
        masked_count = sum(1 for log in logits if log == float("-inf"))
        print_constrained_step(
            token_str,
            logits[token_id],
            masked_count,
            len(logits) - masked_count
        )

        for letter in token_str:
            parameter.append(letter)
            print_progress(letter, "")
            if letter == '"' and (len(parameter) < 2 or parameter[-2] != '\\'):
                break

        if parameter and parameter[-1] in end_tokens:
            break

        loop_counter += 1

    value = "".join(parameter[1:-1]).strip()
    return list(value)


def get_delimiters(parameter_type: ParameterType) -> tuple[str, str]:
    """Return the start and end delimiters for array or object types.

    Args:
        parameter_type: Either ARRAY or OBJECT.

    Returns:
        Tuple of (start_delimiter, end_delimiter).
    """

    if parameter_type == ParameterType.ARRAY:
        return "[", "]"
    return "{\"", "}"


def fallback_number(parameter: list[str],
                    func: Callable[[str], float]) -> list[str]:
    """Return the current parameter if castable, otherwise return ['0'].

    Args:
        parameter: Raw token list including leading quote.
        func: Cast function, either float or int.

    Returns:
        List of characters for the number, or ['0'] on failure.
    """

    try:
        func("".join(parameter[1:]))
        return parameter[1:]
    except ValueError:
        return ["0"]


def fallback_delimited(parameter: list[str],
                       start: str,
                       end: str) -> list[str]:
    """Attempt to repair and return a partial array or object parameter.

    Args:
        parameter: Raw token list built so far.
        start: Opening delimiter.
        end: Closing delimiter.

    Returns:
        List of characters for a valid JSON array or object,
        falling back to '[]' or '{}' if repair fails.
    """

    value = "".join(parameter)

    if not value.endswith(end):
        value += end

    try:
        json.loads(value)
        return list(value)
    except Exception:
        if start == "[":
            return list("[]")

    return list("{}")


def get_recovered_parameter(generated: str, prompt: str) -> str:
    """Attempt to recover a valid parameter value from the original prompt.

    Used when the model generates a string that doesn't appear verbatim
    in the prompt. Tries to extract the value after ':' or handles
    the 'with asterisks' special case.

    Args:
        generated: The string the model generated.
        prompt: The original natural language prompt.

    Returns:
        The recovered parameter string.
    """

    recovered: str = ""
    if ":" in prompt:
        recovered = prompt.split(":", 1)[1].strip()
    elif "with asterisks" in prompt:
        if "*" in generated:
            recovered = "*"
        else:
            recovered = generated
    else:
        recovered = generated

    return recovered

from .shell_prints import print_progress, print_fallback
from .validation_models import ParameterType, FunctionsDefinition
from .llm import BaseLLM
from typing import Callable
import json


MAX_TOKENS = 100
MAX_LOOP_ITER = 50


def get_function_name(valid_func: list[FunctionsDefinition],
                      prompt: str,
                      model: BaseLLM) -> str:
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
            "given in the prompt"
            f"{descriptions}\n"
            f"Prompt: {prompt}\n"
            f"Function to use: {func_name}"
        )

        input_ids = model.encode(starting_string)
        key = tuple(input_ids)

        if key in model.cache:
            logits = model.cache[key].copy()
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            model.cache[key] = logits.copy()

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return functions[0]
            token_id = logits.index(max(logits))

            token_str = (model.clean_function_name(
                model.decode_token(token_id)))

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
        key = tuple(input_ids)

        if key in model.cache:
            logits = model.cache[key].copy()
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            model.cache[key] = logits.copy()

        token_id = logits.index(max(logits))
        token_str = (model.clean_str_tokens(
            model.decode_token(token_id)))

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
        key = tuple(input_ids)

        if key in model.cache:
            logits = model.cache[key].copy()
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            model.cache[key] = logits.copy()

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return fallback_number(parameter, float)

            token_id = logits.index(max(logits))
            token_str = (model.clean_number_tokens(
                model.decode_token(token_id)))

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
        key = tuple(input_ids)

        if key in model.cache:
            logits = model.cache[key].copy()
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            model.cache[key] = logits.copy()

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return fallback_number(parameter, int)

            token_id = logits.index(max(logits))
            token_str = (model.clean_number_tokens(
                model.decode_token(token_id)))

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
        token_str = (model.clean_number_tokens(
            model.decode_token(token_id)))

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
        key = tuple(input_ids)

        if key in model.cache:
            logits = model.cache[key].copy()
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            model.cache[key] = logits.copy()

        token_id = logits.index(max(logits))
        token_str = (model.clean_str_tokens(
            model.decode_token(token_id)))

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
    if parameter_type == ParameterType.ARRAY:
        return "[", "]"
    return "{\"", "}"


def fallback_number(parameter: list[str],
                    func: Callable[[str], float]) -> list[str]:
    try:
        func("".join(parameter[1:]))
        return parameter[1:]
    except ValueError:
        return ["0"]


def fallback_delimited(parameter: list[str],
                       start: str,
                       end: str) -> list[str]:
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
    recovered: str = ""
    if ":" in prompt:
        recovered = prompt.split(":", 1)[1].strip()
    else:
        recovered = generated

    return recovered

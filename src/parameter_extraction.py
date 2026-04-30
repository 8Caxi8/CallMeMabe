from .shell_prints import print_progress, print_fallback
from .validation_models import ParameterType, FunctionsDefinition
from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Callable
import json


MAX_TOKENS = 50
MAX_LOOP_ITER = 50


def get_function_name(valid_func: list[FunctionsDefinition],
                      prompt: str,
                      id_to_token: dict[int, str],
                      model: Small_LLM_Model) -> str:
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

        input_ids = model.encode(starting_string).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return functions[0]
            token_id = logits.index(max(logits))

            token_str = id_to_token.get(
                token_id, "").replace("Ġ", "").replace("ĉ", "")

            candidate = "fn_" + func_name + token_str
            if any(f.startswith(candidate) for f in functions):
                func_name += token_str
                print_progress(token_str, "")
                break

            logits[token_id] = float("-inf")
            tokens_counter += 1

        loop_counter += 1

    return "fn_" + func_name


def get_delimited_parameter(model: Small_LLM_Model,
                            starting_string: str,
                            id_to_token: dict[int, str],
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
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        token_id = logits.index(max(logits))
        token_str = (id_to_token.get(
            token_id, "").replace("Ġ", " ").replace("ĉ", "").
            replace("Ċ", "").replace("ĉ", ""))

        for letter in token_str:
            parameter.append(letter)
            print_progress(letter, "")

            if letter in end_tokens:
                break

        if parameter and parameter[-1] in end_tokens:
            break

        loop_counter += 1

    return parameter


def get_number_parameter(model: Small_LLM_Model,
                         starting_string: str,
                         id_to_token: dict[int, str]) -> list[str]:
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
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return fallback_number(parameter, float)

            token_id = logits.index(max(logits))
            token_str = (id_to_token.get(
                token_id, "").replace("Ġ", "").replace("ĉ", "").
                replace("Ċ", "").replace("ĉ", ""))

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


def get_int_parameter(model: Small_LLM_Model,
                      starting_string: str,
                      id_to_token: dict[int, str]) -> list[str]:
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
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            if tokens_counter > MAX_TOKENS:
                print_fallback(tokens_counter=MAX_TOKENS)
                return fallback_number(parameter, int)

            token_id = logits.index(max(logits))
            token_str = (id_to_token.get(
                token_id, "").replace("Ġ", "").replace("ĉ", "").
                replace("Ċ", "").replace("ĉ", ""))

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


def get_bool_parameter(model: Small_LLM_Model,
                       starting_string: str,
                       id_to_token: dict[int, str]) -> list[str]:
    parameter: list[str] = []
    tokens_counter: int = 0

    input_ids = model.encode(
        starting_string + "".join(
            parameter)).squeeze().tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    while True:
        if tokens_counter > MAX_TOKENS:
            print_fallback(tokens_counter=MAX_TOKENS)
            return ["true"]

        token_id = logits.index(max(logits))
        token_str = (id_to_token.get(
            token_id, "").replace("Ġ", "").replace("ĉ", "").
            replace("Ċ", "").replace("ĉ", ""))

        if token_str.lower() not in {"true", "false"}:
            logits[token_id] = float("-inf")
            tokens_counter += 1
            continue

        parameter.append(token_str)
        print_progress(token_str, "")
        break

    return parameter


def get_string_parameter(model: Small_LLM_Model,
                         starting_string: str,
                         id_to_token: dict[int, str]) -> list[str]:
    parameter: list[str] = ["\""]
    end_tokens = {"\""}
    loop_counter: int = 0

    while True:
        if loop_counter > MAX_LOOP_ITER:
            print_fallback(loop_counter=MAX_LOOP_ITER)
            return parameter[1:] if len(parameter) > 1 else [""]

        input_ids = model.encode(
            starting_string + "".join(
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        token_id = logits.index(max(logits))
        token_str = (id_to_token.get(
            token_id, "").
            replace("Ġ", " ").replace("ĉ", "").
            replace("Ċ", "").replace("ĉ", ""))

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

from .shell_prints import print_progress
from .validation_models import ParameterType
from llm_sdk import Small_LLM_Model  # type: ignore
import inspect


MAX_TOKENS = 50
MAX_LOOP_ITER = 50


class LLMError(Exception):
    def __init__(self, func_name: str, parameter: str, max_iter: int) -> None:
        super().__init__(f"[ERROR]: Passing over {max_iter} iterations!\n"
                         f"   -Function: {func_name}\n"
                         f"   -Parameter: {parameter}\n")


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
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame else "unknown"
            raise LLMError(func_name, "".join(parameter), MAX_LOOP_ITER)

        input_ids = model.encode(
            starting_string + "".join(
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        if max(logits) == float("-inf"):
            break

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
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame else "unknown"
            raise LLMError(func_name, "".join(parameter), MAX_LOOP_ITER)

        input_ids = model.encode(
            starting_string + "".join(
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        if max(logits) == float("-inf"):
            break

        while True:
            if tokens_counter > MAX_TOKENS:
                frame = inspect.currentframe()
                func_name = frame.f_code.co_name if frame else "unknown"
                raise LLMError(func_name, "".join(parameter), MAX_TOKENS)

            token_id = logits.index(max(logits))
            token_str = (id_to_token.get(
                token_id, "").replace("Ġ", "").replace("ĉ", "").
                replace("Ċ", "").replace("ĉ", ""))

            if token_str not in {"."} | end_tokens:
                try:
                    float(token_str)

                except ValueError:
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
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame else "unknown"
            raise LLMError(func_name, "".join(parameter), MAX_LOOP_ITER)

        input_ids = model.encode(
            starting_string + "".join(
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            if tokens_counter > MAX_TOKENS:
                frame = inspect.currentframe()
                func_name = frame.f_code.co_name if frame else "unknown"
                raise LLMError(func_name, "".join(parameter), MAX_TOKENS)

            token_id = logits.index(max(logits))
            token_str = (id_to_token.get(
                token_id, "").replace("Ġ", "").replace("ĉ", "").
                replace("Ċ", "").replace("ĉ", ""))

            if token_str not in end_tokens:
                try:
                    int(token_str)

                except ValueError:
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
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame else "unknown"
            raise LLMError(func_name, "".join(parameter), MAX_TOKENS)

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
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame else "unknown"
            raise LLMError(func_name, "".join(parameter), MAX_LOOP_ITER)

        input_ids = model.encode(
            starting_string + "".join(
                parameter)).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        token_id = logits.index(max(logits))
        token_str = id_to_token.get(
            token_id, "")

        token_str = (id_to_token.get(
            token_id, "").
            replace("Ġ", " ").replace("ĉ", "").
            replace("Ċ", "").replace("ĉ", ""))

        for letter in token_str:
            parameter.append(letter)
            print_progress(letter, "")
            if letter in end_tokens:
                break

        if parameter and parameter[-1] in end_tokens:
            break

        loop_counter += 1

    return parameter[1:-1]


def get_delimiters(parameter_type: ParameterType) -> tuple[str, str]:
    if parameter_type == ParameterType.ARRAY:
        return "[", "]"
    return "{\"", "}"

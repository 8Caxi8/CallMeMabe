from .validation_models import FunctionsDefinition, CallingTests, ParameterType
from .shell_prints import (print_header, print_success_outcome,
                           print_failed_outcome, print_progress,
                           clear_lines)
from .format_data import format_parameters, format_output, FormatError
from .parameter_extraction import (get_bool_parameter, get_delimiters,
                                   get_delimited_parameter,
                                   get_int_parameter, get_number_parameter,
                                   get_string_parameter)
from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Any
import json

lines = 0


class LLMError(Exception):
    def __init__(self, func_name: str, parameter: str, max_iter: int) -> None:
        super().__init__(f"[ERROR]: Passing over {max_iter} iterations!\n"
                         f"   -Function: {func_name}\n"
                         f"   -Parameter: {parameter}\n")


def main_loop(valid_func: list[FunctionsDefinition],
              valid_calls: list[CallingTests]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    model = Small_LLM_Model(device="cpu")
    global lines

    with open(model.get_path_to_vocab_file()) as f:
        vocab = json.load(f)
    id_to_token = {value: key for key, value in vocab.items()}

    for i, call in enumerate(valid_calls):
        prompt = call.prompt
        lines += print_header(i, prompt)

        try:
            func_name = get_function_name(valid_func, prompt,
                                          id_to_token, model)
            function = next(func for func in valid_func
                            if func.name == func_name)

            lines += print_progress("'\n  - 'parameters':", "")
            parameters = get_parameters(function, prompt, id_to_token, model)
            output.append(format_output(prompt, function.name, parameters))

        except ValueError as e:
            lines = clear_lines(lines)
            print_failed_outcome(i, prompt, str(e))
            continue

        except FormatError as e:
            lines = clear_lines(lines)
            print(e)
            continue

        else:
            lines = clear_lines(lines)
            print_success_outcome(output[-1])

    return output


def get_function_name(valid_func: list[FunctionsDefinition],
                      prompt: str,
                      id_to_token: dict[int, str],
                      model: Small_LLM_Model) -> str:
    functions = [f.name for f in valid_func]
    descriptions = "\n".join(
        f"- {function.name.replace('fn_', '')}: {function.description}"
        for function in valid_func
    )

    func_name = ""
    while "fn_" + func_name not in functions:
        starting_string = (
            f"{descriptions}\n"
            f"Prompt: {prompt}\n"
            f"Function to use: {func_name}"
        )

        input_ids = model.encode(starting_string).squeeze().tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        while True:
            token_id = logits.index(max(logits))

            if max(logits) == float("-inf"):
                break

            token_str = id_to_token.get(
                token_id, "").replace("Ġ", "").replace("ĉ", "")

            match_found = False
            for func in functions:
                if "fn_" + func_name + token_str in func:
                    func_name += token_str
                    match_found = True
                    print_progress(token_str, "")
                    break

            if match_found:
                break

            logits[token_id] = float("-inf")

    return "fn_" + func_name


def get_parameters(func: FunctionsDefinition,
                   prompt: str,
                   id_to_token: dict[int, str],
                   model: Small_LLM_Model) -> dict[str, Any]:
    global lines
    parameters_display = "\n".join(
        f"- {key}: {value.type.value}"
        for key, value in func.parameters.items()
    )

    # starting_string = (
    #     f"{parameters_display}\n"
    #     f"Prompt: {prompt}\n"
    #     f"Parameter from prompt (ended with \"):"
    # )

    starting_string = (
        f"{func.description}"
        f"{parameters_display}\n"
        f"Prompt: {prompt}\n"
        f"Parameter (ended with \"):"
    )

    parameters: dict[str, list[str]] = {}

    for param_name, param_def in func.parameters.items():
        parameters[param_name] = []
        parameter_type = param_def.type
        param = f"\n- {param_name} is "

        lines += print_progress(f"\n    - '{param_name}': \"", "")

        try:
            if parameter_type == ParameterType.NUMBER:
                parameters[param_name] = get_number_parameter(
                    model, starting_string + param, id_to_token)

            elif parameter_type == ParameterType.INTEGER:
                parameters[param_name] = get_int_parameter(
                    model, starting_string + param, id_to_token)

            elif parameter_type in (ParameterType.ARRAY, ParameterType.OBJECT):
                parameters[param_name] = get_delimited_parameter(
                    model, starting_string + param, id_to_token,
                    *get_delimiters(parameter_type))

            elif parameter_type == ParameterType.NULL:
                parameters[param_name] = []

            elif parameter_type == ParameterType.BOOLEAN:
                starting_string += "(true/false)"
                parameters[param_name] = get_bool_parameter(
                    model, starting_string + param, id_to_token)

            else:
                parameters[param_name] = get_string_parameter(
                    model, starting_string + param, id_to_token)

        except LLMError as e:
            raise ValueError(str(e))

    return format_parameters(func, parameters)

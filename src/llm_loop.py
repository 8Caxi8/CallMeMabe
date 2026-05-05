from .validation_models import FunctionsDefinition, CallingTests, ParameterType
from .shell_prints import (print_header, print_success_outcome,
                           print_failed_outcome, print_progress,
                           clear_lines, print_recover)
from .format_data import format_parameters, format_output, FormatError
from .parameter_extraction import (get_bool_parameter, get_delimiters,
                                   get_delimited_parameter,
                                   get_int_parameter, get_number_parameter,
                                   get_string_parameter, get_function_name,
                                   get_recovered_parameter)
from .llm import BaseLLM
from typing import Any

lines = 0


def main_loop(valid_func: list[FunctionsDefinition],
              valid_calls: list[CallingTests],
              model: BaseLLM) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    global lines

    for i, call in enumerate(valid_calls):
        prompt = call.prompt
        lines += print_header(i, prompt)

        try:
            func_name = get_function_name(valid_func, prompt,
                                          model)
            function = next(func for func in valid_func
                            if func.name == func_name)

            lines += print_progress("'\n  - 'parameters':", "")
            parameters = get_parameters(function, prompt, model)
            output.append(format_output(prompt, function.name, parameters))

        except FormatError as e:
            lines = clear_lines(lines)
            print_failed_outcome(i, prompt, str(e))
            continue

        except Exception as e:
            lines = clear_lines(lines)
            print_failed_outcome(i, prompt, str(e))
            continue

        else:
            lines = clear_lines(lines)
            print_success_outcome(output[-1])

    return output


def get_parameters(func: FunctionsDefinition,
                   prompt: str,
                   model: BaseLLM) -> dict[str, Any]:
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
        "Your task is to COPY text, not to generate or rewrite.\n"
        "Copy the parameter EXACTLY as it appears in the prompt.\n"
        "Do NOT remove anything.\n"
        "Character-by-character copy.\n"
        f"Parameter (start and end with \"):"
    )

    parameters: dict[str, list[str]] = {}

    for param_name, param_def in func.parameters.items():
        parameters[param_name] = []
        parameter_type = param_def.type
        param = f"\n- {param_name} is "

        lines += print_progress(f"\n    - '{param_name}': \"", "")

        if parameter_type == ParameterType.NUMBER:
            parameters[param_name] = get_number_parameter(
                model, starting_string + param)

        elif parameter_type == ParameterType.INTEGER:
            parameters[param_name] = get_int_parameter(
                model, starting_string + param)

        elif parameter_type in (ParameterType.ARRAY, ParameterType.OBJECT):
            parameters[param_name] = get_delimited_parameter(
                model, starting_string + param,
                *get_delimiters(parameter_type))

        elif parameter_type == ParameterType.NULL:
            parameters[param_name] = []

        elif parameter_type == ParameterType.BOOLEAN:
            parameters[param_name] = get_bool_parameter(
                model, starting_string + "(true/false)" + param)

        else:
            generated = "".join(get_string_parameter(
                model, starting_string + param))
            if generated not in prompt:
                recovered = get_recovered_parameter(generated, prompt)
                if recovered != generated:
                    lines += print_recover("[WARNING]: Recovering llm output"
                                           f" to '{recovered}'...", "")
                parameters[param_name] = list(recovered)

            else:
                parameters[param_name] = list(generated)

    return format_parameters(func, parameters)

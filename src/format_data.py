from .validation_models import FunctionsDefinition, ParameterType
from typing import Any
import json


class FormatError(Exception):
    pass


def format_parameters(func: FunctionsDefinition,
                      parameters: dict[str, list[str]]) -> dict[str, Any]:
    formated_parameters: dict[str, Any] = {}
    for para_name, para_type in func.parameters.items():
        value = parameters.get(para_name, [])
        if para_type.type == ParameterType.STRING:
            string = "".join(value)
            formated_parameters[para_name] = string
        elif para_type.type == ParameterType.NUMBER:
            number = float("".join(value))
            formated_parameters[para_name] = number
        elif para_type.type == ParameterType.INTEGER:
            number = int("".join(value))
            formated_parameters[para_name] = number
        elif para_type.type == ParameterType.BOOLEAN:
            boolean = "".join(value).lower()

            if boolean == "true":
                formated_parameters[para_name] = True
            elif boolean == "false":
                formated_parameters[para_name] = False
            else:
                raise FormatError(f"[ERROR]: Invalid boolean {boolean}")

        elif para_type.type == ParameterType.ARRAY:
            formated_parameters[para_name] = json.loads("".join(value))
        elif para_type.type == ParameterType.NULL:
            formated_parameters[para_name] = None
        elif para_type.type == ParameterType.OBJECT:
            formated_parameters[para_name] = json.loads("".join(value))
        else:
            raise FormatError("[ERROR]: Unsupported parameter type: "
                              f"{para_type.type}")

    return formated_parameters


def format_output(prompt: str, function: str,
                  parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "name": function,
        "parameters": parameters,
    }

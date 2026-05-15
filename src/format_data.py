from .validation_models import FunctionsDefinition, ParameterType
from typing import Any
import json


class FormatError(Exception):
    """Raised when a parameter value cannot be cast to its expected type."""
    pass


def format_parameters(func: FunctionsDefinition,
                      parameters: dict[str, list[str]]) -> dict[str, Any]:
    """Convert raw token lists into typed Python values.

    Args:
        func: The validated function definition containing parameter types.
        parameters: Mapping of parameter names to their raw token lists.

    Returns:
        Mapping of parameter names to their typed Python values.

    Raises:
        FormatError: If a value cannot be cast to its declared type.
    """
    formated_parameters: dict[str, Any] = {}
    for para_name, para_type in func.parameters.items():
        value = parameters.get(para_name, [])
        if para_type.type == ParameterType.STRING:
            string = "".join(value)
            formated_parameters[para_name] = string
        elif para_type.type == ParameterType.NUMBER:
            try:
                number = float("".join(value))
            except ValueError as e:
                raise FormatError(e)
            formated_parameters[para_name] = number
        elif para_type.type == ParameterType.INTEGER:
            try:
                number = int("".join(value))
            except ValueError as e:
                raise FormatError(e)
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
            try:
                formated_parameters[para_name] = json.loads("".join(value))
            except ValueError:
                formated_parameters[para_name] = json.loads("[]")

        elif para_type.type == ParameterType.NULL:
            formated_parameters[para_name] = None
        elif para_type.type == ParameterType.OBJECT:
            try:
                formated_parameters[para_name] = json.loads("".join(value))
            except ValueError:
                formated_parameters[para_name] = json.loads("{}")
        else:
            raise FormatError("[ERROR]: Unsupported parameter type: "
                              f"{para_type.type}")

    return formated_parameters


def format_output(prompt: str, function: str,
                  parameters: dict[str, Any]) -> dict[str, Any]:
    """Assemble the final output dict for a single function call.

    Args:
        prompt: The original natural language prompt.
        function: The selected function name.
        parameters: The typed parameter values.

    Returns:
        Dict with keys 'prompt', 'name', and 'parameters'.
    """
    return {
        "prompt": prompt,
        "name": function,
        "parameters": parameters,
    }

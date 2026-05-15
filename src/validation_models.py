from pydantic import BaseModel, model_validator, ValidationError
from pydantic_core import PydanticCustomError as Pe
from enum import Enum
from typing import Any
from .llm import BaseLLM, Qwen3LLM, Qwen2LLM
import sys


class CallingTests(BaseModel):
    """Pydantic model for validating a single prompt entry from the
    input file.
    """
    prompt: str


class ParameterType(str, Enum):
    """Supported parameter types for function definitions."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class Parameter(BaseModel):
    """Pydantic model for a single function parameter with its type."""
    type: ParameterType


class FunctionsDefinition(BaseModel):
    """Pydantic model for a complete function definition.

    Validates that the function name starts with 'fn_' and that
    all parameters and return type use supported ParameterTypes.
    """
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter

    @model_validator(mode='after')
    def validate_function(self) -> "FunctionsDefinition":
        """Validate that the function name starts with 'fn_'.

        Raises:
            PydanticCustomError: If the name does not start with 'fn_'.
        """
        if not self.name.startswith("fn_"):
            raise Pe("starting_name_error",
                     f"Function name {self.name} must start with: 'fn_'")

        return self


def get_validated_model(functions: list[dict[str, Any]],
                        input_file: list[dict[str, Any]],
                        llm: str) \
            -> tuple[list[FunctionsDefinition], list[CallingTests], BaseLLM]:
    """Validate all function definitions and prompts, and initialize the LLM.

    Exits with code 2 if any function definition or prompt fails validation,
    or if the requested LLM name is not recognized.

    Args:
        functions: Raw list of function definition dicts from the JSON file.
        input_file: Raw list of prompt dicts from the input JSON file.
        llm: The model name to initialize, either 'qwen3' or 'qwen2'.

    Returns:
        Tuple of (validated_functions, validated_calls, model).
    """

    validated_calls: list[CallingTests] = []
    validated_functions: list[FunctionsDefinition] = []
    model: BaseLLM

    for function in functions:
        try:
            validated_functions.append(
                FunctionsDefinition.model_validate(function))
        except ValidationError as e:
            print("[Error]: Validation failed for "
                  f"{function.get('name', 'unknown')}:")
            for error in e.errors():
                print(f"      {error['msg']}: {error['loc']}")
                print(f"      Instead got:{error['input']}")
            sys.exit(2)

    for call in input_file:
        try:
            validated_calls.append(
                CallingTests.model_validate(call))
        except ValidationError as e:
            print("[Error]: Validation failed:")
            for error in e.errors():
                print(f"      {error['msg']}: {error['loc']}")
                print(f"      Instead got:{error['input']}")
            sys.exit(2)

    if llm == "qwen3":
        model = Qwen3LLM(device="cpu")

    elif llm == "qwen2":
        model = Qwen2LLM()

    else:
        print(f"[ERROR]: Unkown llm {llm}!")
        sys.exit(2)

    return validated_functions, validated_calls, model

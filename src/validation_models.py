from pydantic import BaseModel, model_validator, ValidationError
from pydantic_core import PydanticCustomError as Pe
from enum import Enum
from typing import Any
import sys


class CallingTests(BaseModel):
    prompt: str 


class ParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class Parameter(BaseModel):
    type: ParameterType


class FunctionsDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter

    @model_validator(mode='after')
    def validate_function(self) -> "FunctionsDefinition":
        if not self.name.startswith("fn_"):
            raise Pe("starting_name_error",
                     f"Function name {self.name} must start with: 'fn_'")

        return self


def get_validated_model(functions: list[dict[str, Any]],
                        input_file: list[dict[str, Any]]) \
                    -> tuple[list[FunctionsDefinition], list[CallingTests]]:
    validation = True
    validated_calls: list[CallingTests] = []
    validated_functions: list[FunctionsDefinition] = []

    for function in functions:
        try:
            validated_functions.append(
                FunctionsDefinition.model_validate(function))
        except ValidationError as e:
            print("[Error]: Validation failed for "
                  f"{function.get('name', 'unknown')}:")
            for error in e.errors():
                print(f"      {error['msg']}: {error['loc']}")
            validation = False

    for call in input_file:
        try:
            validated_calls.append(
                CallingTests.model_validate(call))
        except ValidationError as e:
            print("[Error]: Validation failed:")
            for error in e.errors():
                print(error["msg"])
            validation = False

    if not validation:
        sys.exit(2)

    return validated_functions, validated_calls

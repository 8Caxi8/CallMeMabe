from pydantic import BaseModel, Field, model_validator, ValidationError
from pydantic_core import PydanticCustomError as Pe
import sys


class CallingTests(BaseModel):
    prompt: str = Field(min_length=3)


class Parameter(BaseModel):
    type: str

    @model_validator(mode='after')
    def unknown_type(self) -> "Parameter":
        known = {"string", "number", "boolean", "integer",
                 "array", "object", "null"}
        if self.type not in known:
            sys.stderr.write(f"[WARNING]: Unknown type '{self.type}'\n")

        return self


class FunctionsDefinition(BaseModel):
    name: str = Field(min_length=4)
    description: str = Field(min_length=4)
    parameters: dict[str, Parameter]
    returns: Parameter

    @model_validator(mode='after')
    def validate_function(self) -> "FunctionsDefinition":
        if not self.name.startswith("fn_"):
            raise Pe("starting_name_error",
                     f"Function name {self.name} must start with: 'fn_'")

        return self


def get_validated_model(functions: list[dict], input_file: list[dict]) \
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

from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError as Pe
from enum import Enum


class Type(str, Enum):
    STRING = "string"
    NUMBER = "number"
    LIST = "list"
    DICT = "dictionary"


class CallingTests(BaseModel):
    prompt: str = Field(min_length=3, max_length=50)


class Parameter(BaseModel):
    type: Type


class FunctionsDefinition(BaseModel):
    name: str = Field(min_length=4, max_length=50)
    description: str = Field(min_length=4, max_length=200)
    parameters: dict[str, Parameter]
    returns: Parameter

    @model_validator(mode='after')
    def validate_function(self) -> "FunctionsDefinition":
        if not self.name.startswith("fn_"):
            raise Pe("starting_name_error",
                     f"Function name {self.name} must start with: 'fn_'")

        return self

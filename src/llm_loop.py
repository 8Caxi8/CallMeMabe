from .validation_models import FunctionsDefinition, CallingTests, ParameterType
from .shell_prints import print_header, print_outcome
from .format_data import format_parameters, format_output
from llm_sdk import Small_LLM_Model
import json
from typing import Any


def main_loop(valid_func: list[FunctionsDefinition],
              valid_calls: list[CallingTests]) -> list[dict]:
    output: list[dict] = []
    model = Small_LLM_Model(device="cpu")

    with open(model.get_path_to_vocab_file()) as f:
        vocab = json.load(f)
    id_to_token = {value: key for key, value in vocab.items()}

    for i in range(len(valid_calls)):
        prompt = valid_calls[i].prompt
        print_header(prompt)

        func_name = get_function_name(valid_func, prompt, id_to_token, model)
        function = next(func for func in valid_func if func.name == func_name)

        parameters = get_parameters(function, prompt, id_to_token, model)
        output.append(format_output(prompt, function.name, parameters))

        print_outcome(output[i])

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
            "Choosing one from the following functions:\n"
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
                    break
            if match_found:
                break
            logits[token_id] = float("-inf")

    return "fn_" + func_name


def get_parameters(func: FunctionsDefinition,
                   prompt: str,
                   id_to_token: dict[int, str],
                   model: Small_LLM_Model) -> dict[str, Any]:
    parameters_display = "\n".join(
        f"- {key}: {value.type}"
        for key, value in func.parameters.items()
    )

    starting_string = (
        "Extract the parameters from the prompt:\n"
        f"Function to use: {func.name}\n"
        f"{parameters_display}\n"
        f"Prompt: {prompt}\n"
        f"parameters:"
    )
    parameters_list = [key for key in func.parameters.keys()]
    parameters: dict[str, list[str]] = {}

    i = 0
    while i < len(parameters_list):
        parameters.update({parameters_list[i]: []})

        if func.parameters[parameters_list[i]].type == ParameterType.ARRAY:
            parameters[parameters_list[i]].append("[")
            starting_string += f"\n- {parameters_list[i]}: "
            end_tokens = {"]"}
        else:
            starting_string += f"\n- {parameters_list[i]}: '"
            end_tokens = {"'", "\""}

        while True:
            input_ids = model.encode(
                starting_string + "".join(
                    parameters[parameters_list[i]])).squeeze().tolist()
            logits = model.get_logits_from_input_ids(input_ids)
            print(starting_string + "".join(
                    parameters[parameters_list[i]]))

            if max(logits) == float("-inf"):
                break

            while True:
                token_id = logits.index(max(logits))
                token_str = id_to_token.get(
                    token_id, "").replace("Ġ", " "
                                          ).replace("ĉ", "").replace("Ċ", "")

                if token_str in end_tokens:
                    break

                if func.parameters[parameters_list[i]].type == ParameterType.NUMBER or \
                   func.parameters[parameters_list[i]].type == ParameterType.INTEGER and \
                   token_str not in {".", ","}:
                    try:
                        float(token_str)
                    except ValueError:
                        logits[token_id] = float("-inf")
                        continue

                parameters[parameters_list[i]].append(token_str)
                break

            if token_str in end_tokens:
                break

        if func.parameters[parameters_list[i]].type == ParameterType.ARRAY:
            starting_string += "".join(
                    parameters[parameters_list[i]]) + "]"
            parameters[parameters_list[i]].append("]")
        else:
            starting_string += "".join(
                    parameters[parameters_list[i]]) + "'"
        print(starting_string)
        print(parameters[parameters_list[i]])
        i += 1

    return format_parameters(func, parameters)


# top_5 = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
# for token_id in top_5:
#     token_str = id_to_token.get(token_id, "")
#     print(f"Token ID: {token_id} | Score: {logits[token_id]:.4f} |
#       String: '{token_str}'")

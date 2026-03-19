from .validation_models import FunctionsDefinition, CallingTests
from llm_sdk import Small_LLM_Model
import json


def main_loop(valid_func: list[FunctionsDefinition],
              valid_calls: list[CallingTests]) -> None:
    model = Small_LLM_Model(device="cpu")
    prompt =  valid_calls[9].prompt

    with open(model.get_path_to_vocab_file()) as f:
        vocab = json.load(f)
    id_to_token = {value: key for key, value in vocab.items()}

    func_name = get_function_name(valid_func,
                                  prompt, id_to_token,
                                  model)

    func = next(func for func in valid_func if func.name == func_name)

    parameters = "\n".join(
        f"- {key}: {value.type}"
        for key, value in func.parameters.items()
    )

    starting_string = (
        "Choosing the correct input:\n"
        f"Function to use: {func_name}\n"
        f"{parameters}\n"
        f"Prompt: {prompt}\n"
        f"parameters:"
    )
    parameters_list = [key for key in func.parameters.keys()]
    i = 0

    starting_string += f"\n- {parameters_list[i]}: "

    input_ids = model.encode(starting_string).squeeze().tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    print(starting_string)

    top_5 = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
    for token_id in top_5:
        token_str = id_to_token.get(token_id, "")
        print(f"Token ID: {token_id} | Score: {logits[token_id]:.4f} | String: '{token_str}'")

    """ while i < len(func.parameters):
        starting_string +  """

    print(func_name)


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

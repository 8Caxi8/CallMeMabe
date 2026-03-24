*This project has been created as part of the 42 curriculum by dansimoe*

# Description

The objective of this project is to explore how **Large Language Models** (*LLMs*) work and how they can be used to transform natural language into structured, machine-executable outputs.

In this project we will use a relatively small language model, the **Qwen3-0.6B**, which is a compact, open-source causal language model developed by Alibaba Cloud, with only 600 million parameters. Compared to other **7B+** models, **Qwen3-0.6B** is significantly faster, capable of running on low-power, consumer edge hardware, or in local environments where **7B+** models cannot fit.

On the other hand, due to the limited size of the model, constrained decoding is applied to ensure reliable and structured outputs, producing a **>90%** accuracy for a set of prompts and a **100%** accuracy for `JSON` generated files as the output for the function calls.

Next is an example of a function call output for a given prompt:
```bash
[
	{
		"prompt": "What is the sum of 2 and 3?",
		"name": "fn_add_numbers",
		"parameters": {"a": 2.0, "b": 3.0}
	}
]
```

# Instructions

All the relevant commands are the makefile rules:
```bash
make install # Installs the VM with its required dependencies
```
```bash
make run # Runs the program with its defaults input and output files
```
```bash
make debug # Runs the program in a debug mode
```
```bash
make clean # Clean the __pycache__, .mypy_cache and .*pyc files
```
```bash
make lint # Executes the flake8 and mypy validation
```
```bash
make lint-strict # Executes the flake8 and mypy in strict mode validation
```

To change the paths of the files just add:
- --input <input_file>
- --output <output_file>
- --functions_definition <function_definition_file>

and run:
```bash
uv run python -m src [--functions_definition <...> ...]
```

# Code
## Algorithm Explanation
- To guarantee **100%** valid `JSON` output, I apply constrained decoding to all semantic elements (function name and parameters), and then assemble the final `JSON` structure deterministically. This avoids structural errors from the language model.

So in short for each prompt I loop for:
```bash
	get_function_name()
	get_parameters()
	format_output()
	output.append(result)
```
where output is a `list` and the result is the `dict` with each key being the `prompt`, `name` and the `parameters`.

- If any error is encountered when getting any of the elements or in the format section, I discard this result and move to the next.


### Constrained Decoding Strategy
At each generation step:
- The model produces logits for all tokens
- Invalid tokens (those not matching the expected structure or type) are masked by setting their logits to `-inf`
- The next token is selected only from the remaining valid tokens

This guarantees that:
- Function names always match available definitions
- Parameter values respect their expected types
- Invalid tokens are never selected

---

This approach ensures that all generated outputs are both semantically valid (through constrained decoding) and structurally valid (through deterministic formatting).

## Design Decisions
- The biggest decision was to use the llm just to output the function name and the parameters extracted from the prompt.

- Also, I used the `pydantic` model to ensure each of the functions were well defined with all the parameters and correct types for a `JSON` file. This also ensure the main loop was working, since for example I had to use the function descriptions for the llm usage.

- So the key for using the `llm_sdk` package was to give the model a detailed message of was requested, in the case of the `get_function_name`, the initial message given to the model is:
```bash
starting_string = (
            f"{descriptions}\n" 			# funcion.name: function.description
            f"Prompt: {prompt}\n" 			# prompt used
            f"Function to use: {func_name}" # being guessed by the llm
        )
```
This message was passed to llm each time I appended a token, to ensure the next highest value token was the correct one. To ensure the function was a valid function, I only accepted tokens if they correctly described a possible function name from the available functions.

- For the `get_parameters`the initial message used is:
```bash
starting_string = (
        f"{parameters_display}\n"					# parameters: parameters type
        f"Prompt: {prompt}\n"						# prompt used
        f"Parameter from prompt (ended with \"):"	# each parameter to be guessed by llm
    )
```
Notice for example the need to `(ended with \")` So the llm knows it needs to end the guessed parameter with the `"` token, and I use this to stop the llm loop.

- Although the prompt is used to guide the model semantically, I enforce correctness through constrained decoding rather than relying  on the model to produce valid outputs autonomously.

- For each parameter, a type-specific getter is called:
	- `STRING` - generates until closing `"`.
	- `NUMBER` - only allows numeric tokens and `.`, generates until closing `"`
	- `INTEGER` - only allows numeric tokens without `.`, generates until closing `"`
	- `ARRAY` - uses `[`/`]` as delimiters, letter-by-letter scanning
	- `OBJECT` - uses `{`/`}` as delimiters, same approach as array
	- `BOOLEAN` - constrains output to only `true`/`false` tokens
	- `NULL` - returns `None` directly, no generation needed

## Performance Analysis
- For the speed, the model was relatively fast for guessing the solutions. As for accuracy, I needed several tests, with several initial `starting_string` to ensure the model was guessing right. The function name was easy to retrieve, but the parameters were more tricky. 

- Boolean extraction proved challenging due to model bias toward `"true"`. This highlights limitations of small models and the importance of stronger constraints in such cases. I even added a `(true/false)` statement at the beginning of guessing the right input, but the highest token is always `true` even when I give it a false in the prompt to help it guess. So if I wanted to use make this more reliable in this section I should add a set of negative tokens and help the model to use this set to see if in the prompt any keywords were in this set, and in this manner it should look for negative keywords and return `false`. But as stated in the subject I should use the `llm`, not *'with heuristics or any other sort of medieval magic'*

## Challenges Faced
- The challenges I had were mainly in the getting the correct parameters from the prompt. I ultimately used a simple starting string for each parameter, where I used a display of each parameter, the prompt next (and the order matters, since the llm will have this prompt close to the guessed answer), and the parameter to be guessed. Here I also filter the parameter to be guessed by the `llm` if there is more than one. So, for example fo the regex function:
```bash
[
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {
        "type": "string"
      },
      "regex": {
        "type": "string"
      },
      "replacement": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
```
what the llm sees when is guessing each parameter is:

[![asciicast](https://asciinema.org/a/25dgE7vipDSQ8FZw.svg)](https://asciinema.org/a/25dgE7vipDSQ8FZw)

- Special Tokens (like `Ġ` or `Ċ`) were normalized and used when needed.

- This model also is known for infinite loops, and being repeatedly generating the same tokens, so I solved this using maximum iteration limits.

## Testing Strategy
- I tested with the provided 11-prompt, which I get near **100%** accuracy for function selection and argument extraction.
- I created more input for testing other types like `object` type, `boolean` type and `int` type. When testing the different types I realized I should have different functions for extracting this types, and this were created at `parameter_extraction.py`.
- For debugging and understanding what was being selected, some print custom functions were created to create print at running time to the prompt what was being selected as well if any errors had occured:

## Example Usage

```bash
# run with default files
make run

# run with custom files
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

[![asciicast](https://asciinema.org/a/wtzUeFp3dzOjSPRR.svg)](https://asciinema.org/a/wtzUeFp3dzOjSPRR)


## Resources

- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [JSON Schema Types](https://json-schema.org/understanding-json-schema/reference/type)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [LLM Explained](https://www.youtube.com/watch?v=LPZh9BOjkQs&t=9s)

## AI Usage
- Claude was used to:
	- Debugging token generation logic
	- Code review for main llm loop
	- Prompt engineering for parameter extraction at the parameter extration function

*This project has been created as part of the 42 curriculum by dansimoe*

# Description

The objective of this project is to explore how **Large Language Models** (*LLMs*) work and how they can be used to transform natural language into structured, machine-executable outputs.

In this project we will use a relatively small language model, the **Qwen3-0.6B**, which is a compact, open-source causal language model developed by Alibaba Cloud, with only 600 million parameters. Compared to other **7B+** models, **Qwen3-0.6B** is significantly faster, capable of running on low-power, consumer edge hardware, or in local environments where **7B+** models cannot fit.

On the other hand, due to the limited size of the model, constrained decoding is applied to ensure reliable and structured outputs, producing a **>90%** accuracy for a set of prompts and a **100%** accuracy for `JSON` generated files as the output for the function calls.

Next is an example of a function call output for a given prompt:
```json
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
make install     # Installs the virtual environment with its required dependencies
make run         # Runs the program with its default input and output files
make debug       # Runs the program with Python's pdb debugger in verbose mode
make verbose     # Runs the program with verbose mode showing constrained decoding steps
make clean       # Cleans __pycache__, .mypy_cache and .pyc files
make lint        # Executes flake8 and mypy validation
make lint-strict # Executes flake8 and mypy in strict mode
```

To change the paths of the files or select a different model:
```bash
uv run python -m src \
  [--functions_definition <function_definition_file>] \
  [--input <input_file>] \
  [--output <output_file>] \
  [--llm <qwen3|qwen2>] \
  [--verbose]
```

**Arguments:**
- `--input` — path to the JSON file containing prompts (default: `data/input/function_calling_tests.json`)
- `--output` — path for the output JSON file (default: `data/output/function_calling_results.json`)
- `--functions_definition` — path to the functions definition JSON file (default: `data/input/functions_definition.json`)
- `--llm` — model to use, either `qwen3` (default) or `qwen2`
- `--verbose` — enable debug mode showing constrained decoding steps (token selected, logit score, masked count)

# Code

## Algorithm Explanation

To guarantee **100%** valid `JSON` output, constrained decoding is applied to all semantic elements (function name and parameters), and then the final `JSON` structure is assembled deterministically. This avoids structural errors from the language model.

For each prompt the loop runs:
```
get_function_name()
get_parameters()
format_output()
output.append(result)
```
where `output` is a `list` and each result is a `dict` with keys `prompt`, `name`, and `parameters`.

If any error is encountered during extraction or formatting, that result is discarded and the loop moves to the next prompt.

### Constrained Decoding Strategy

At each generation step:
- The model produces logits for all tokens in the vocabulary
- Invalid tokens (those not matching the expected structure or type) are masked by setting their logits to `-inf`
- The next token is selected only from the remaining valid tokens

This guarantees that:
- Function names always match one of the available definitions
- Parameter values respect their expected types
- Invalid tokens are never selected

## Design Decisions

The biggest decision was to use the LLM only to output the function name and extract the parameters from the prompt — not to generate any free-form text.

`pydantic` models are used to validate each function definition, ensuring all parameters have correct types before the main loop runs.

### Function Name Selection

The starting string given to the model for function name selection is:

```python
starting_string = (
    "You have to select the correct function by what is "
    "given in the prompt.\n"
    "Select a function name by keywords in the prompt "
    "EXACTLY as they are written.\n"
    f"{descriptions}\n"        # - function_name: description
    f"Prompt: {prompt}\n"
    f"Function to use: {func_name}"  # being built token by token
)
```

This message is re-submitted to the model each time a token is appended, so the model always has the full context of what has been generated so far. A token is only accepted if it extends a valid function name prefix — otherwise its logit is set to `-inf` and the next best token is tried.

### Parameter Extraction

The starting string for parameter extraction is:

```python
starting_string = (
    f"{func.description}"
    f"{parameters_display}\n"       # - param_name: param_type
    f"Prompt: {prompt}\n"
    "Your task is to COPY text, not to generate or rewrite.\n"
    "Copy the parameter EXACTLY as it appears in the prompt.\n"
    "Do NOT remove anything.\n"
    "Character-by-character copy.\n"
    f"Parameter (start and end with \"):"
)
```

The `(ended with \")` instruction tells the model it must close the parameter with a `"` token, which is used as the stop condition for the generation loop. Each parameter is extracted with a type-specific getter:

- `STRING` — generates until closing `"`
- `NUMBER` — only allows numeric tokens and `.` and `-`, generates until closing `"`
- `INTEGER` — only allows numeric tokens and `-` without `.`, generates until closing `"`
- `ARRAY` — uses `[`/`]` as delimiters, letter-by-letter scanning
- `OBJECT` — uses `{"` /`}` as delimiters, same approach as array
- `BOOLEAN` — constrains output strictly to `true`/`false` tokens
- `NULL` — returns `None` directly, no generation needed

Although the prompt guides the model semantically, correctness is enforced through constrained decoding rather than relying on the model to produce valid outputs autonomously.

## Bonus Features

### Multi-model Support
The project supports two models via the `--llm` flag:
- `qwen3` (default) — Qwen/Qwen3-0.6B, better accuracy
- `qwen2` — Qwen/Qwen2-0.5B, faster but lower accuracy

Both models go through the `llm_sdk` package, with model-specific token cleaning to handle the different special characters each tokenizer uses (`Ġ`, `Ċ` for Qwen3 vs `Ġ`, `▁`, `\n` for Qwen2).

### Verbose Debug Mode
Running with `--verbose` (or `make debug`) reveals the full constrained decoding process at each step:

```
[CONSTRAINED] selected: 'add' (logit: 21.96) | masked: 0 | valid candidates: 151936
[CONSTRAINED] selected: '_numbers' (logit: 29.69) | masked: 0 | valid candidates: 151936
```

This shows which token was selected, its logit score, how many tokens were masked as invalid, and how many valid candidates remained. This is particularly useful for understanding why the model makes certain selections and for debugging edge cases.

### Token Caching
All decoded tokens are cached in `BaseLLM` via `get_cached_token()`, avoiding redundant decode calls for the same token ID across generation steps. This significantly reduces processing time for longer sequences.

### Partial Tokenizer Implementation
The vocabulary JSON file (`get_path_to_vocab_file()`) is loaded manually and used to implement custom `decode_token()` — returning token strings by ID without relying on the SDK's decode method. This gives full control over token normalization and the special character cleaning applied per model.

## Performance Analysis

Qwen3-0.6B achieves near **100%** accuracy on the standard test set and performs well on most edge cases. Qwen2-0.5B performs noticeably worse, particularly on function selection for ambiguous prompts.

**Known limitations and edge cases:**
- Unconventional number formats like `-.5` or `.25` (leading decimal without zero) — the model picks the digit token directly, missing the sign or decimal prefix
- Empty string arguments — when the prompt contains an empty string (e.g. `Reverse ''`), the model always generates something rather than an empty parameter
- Completely unrelated prompts (e.g. `Greet @#$%`) — defaults to the first available function
- Boolean extraction is biased toward `true` — the model consistently selects `true` as the highest-probability token even when the prompt suggests `false`
- Scientific notation like `1e10` is not fully supported — extracts `10` instead of `1e10`

For speed, the model processes the standard 11-prompt test set in under a minute on CPU. Larger test sets with complex regex patterns take longer due to the token-by-token generation approach.

## Challenges Faced

The main challenges were in getting correct parameters from the prompt. The final solution uses a structured starting string that places the prompt immediately before the parameter being guessed, so the model has maximum context when making its selection.

For the regex function with three parameters (`source_string`, `regex`, `replacement`), the model sees a different starting string for each parameter, with the parameter name appended at the end:

[![asciicast](https://asciinema.org/a/25dgE7vipDSQ8FZw.svg)](https://asciinema.org/a/25dgE7vipDSQ8FZw)

Special tokens like `Ġ` (space prefix) and `Ċ` (newline) needed to be normalized differently per model, since Qwen3 and Qwen2 use different tokenizer conventions.

Infinite loops were a recurring issue — the model would repeatedly generate the same token when stuck. This was solved with `MAX_LOOP_ITER` and `MAX_TOKENS` limits, with fallback values returned when limits are hit.

## Testing Strategy

Testing was done with multiple input files covering different scenarios:

- `function_calling_tests.json` — the standard 11-prompt test set, near 100% accuracy
- `edge_cases.json` — 38 edge cases including empty prompts, garbage input, missing arguments, and unusual number formats
- `hard_prompts.json` — ambiguous prompts where the function is implied but not named explicitly
- `regex_tests.json` — regex-specific prompts testing pattern extraction
- `tests2.json` — extended general tests including palindrome and boolean checks
- `function_calling_moulinette.json` — prompts matching the likely evaluator test set
- `object.json` — object parameter extraction tests
- `functions_advanced.json` — tests with boolean, array, and object parameter types

The verbose debug mode was used throughout to inspect which tokens were being selected and why, making it straightforward to identify and fix prompt engineering issues.

## Example Usage

```bash
# Run with default files
make run

# Run with verbose debug output showing constrained decoding steps
make verbose

# Run with pdb debugger
make debug

# Run with custom files and Qwen2 model
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json \
  --llm qwen2

# Run with verbose mode
uv run python -m src --verbose
```

[![asciicast](https://asciinema.org/a/wtzUeFp3dzOjSPRR.svg)](https://asciinema.org/a/wtzUeFp3dzOjSPRR)

## Resources

- [Qwen3-0.6B Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen2-0.5B Model](https://huggingface.co/Qwen/Qwen2-0.5B)
- [JSON Schema Types](https://json-schema.org/understanding-json-schema/reference/type)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [LLM Explained](https://www.youtube.com/watch?v=LPZh9BOjkQs&t=9s)
- [Constrained Decoding Overview](https://huggingface.co/blog/constrained-beam-search)

## AI Usage

AI was used to:
- Debugging the `Qwen2LLM` import.
- Identifying edge cases and known limitations from test output analysis

import sys
from .parser import parser, output_json
from .validation_models import get_validated_model
from .llm_loop import main_loop
from .shell_prints import print_output, set_verbose


def main() -> None:
    """Entry point for the function calling tool.

    Parses CLI arguments, initializes the LLM model, runs the main
    generation loop over all prompts, and writes the results to the
    output JSON file. Exits with code 1 on argument or unexpected errors.
    """
    try:
        functions, input_file, output_file_path, llm, verbose = parser()
        set_verbose(verbose)

    except ValueError as e:
        print(e)
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    output = main_loop(*get_validated_model(functions, input_file, llm))

    try:
        print_output(output_file_path)
        output_json(output, output_file_path)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()

import sys
from .parser import parser, output_json
from .validation_models import get_validated_model
from .llm_loop import main_loop
from .shell_prints import print_llm_initializer


def main() -> None:
    try:
        functions, input_file, output_file_path = parser()

    except ValueError as e:
        print(e)
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_llm_initializer(len(input_file), len(functions))
    output = main_loop(*get_validated_model(functions, input_file))

    try:
        output_json(output, output_file_path)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()

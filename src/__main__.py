import sys
from .parser import parser, validate_json


def main() -> None:

    try:
        functions, input_file, output_file = parser()
        validate_json(functions, input_file)

    except ValueError as e:
        print(e)
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

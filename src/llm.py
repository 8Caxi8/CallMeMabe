import json
from llm_sdk import Small_LLM_Model  # type: ignore
from abc import ABC, abstractmethod
from typing import Callable, cast
from .shell_prints import print_llm_initializer


class BaseLLM(ABC):
    """Abstract base class for LLM wrappers."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Initialize the SDK model, load weights and vocabulary.

        Args:
            model_name: The HuggingFace model identifier to load.
            device: The compute device to use, either 'cpu', 'cuda', or 'mps'.
        """
        self.cache: dict[tuple[int, str], str] = {}
        self._model = Small_LLM_Model(model_name=model_name, device=device)
        self._vocab = self._load_vocab()

    def _load_vocab(self) -> dict[int, str]:
        """Load the vocabulary file and return an id-to-token mapping."""
        vocab_path = self._model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            token_to_id: dict[str, int] = json.load(f)
        return {v: k for k, v in token_to_id.items()}

    def get_cached_token(self,
                         token_id: int,
                         clean_fn: Callable[[str], str]) -> str:
        """Decode a token ID and apply a cleaning function, with caching.

        Args:
            token_id: The token ID to decode.
            clean_fn: The cleaning function to apply to the decoded token.

        Returns:
            The cleaned token string, cached for future calls.
        """
        key = (token_id, clean_fn.__name__)
        if key not in self.cache:
            self.cache[key] = clean_fn(self.decode_token(token_id))
        return self.cache[key]

    def encode(self, text: str) -> list[int]:
        """Encode text using the SDK tokenizer, returning a flat list of IDs.
        """
        return cast(list[int], self._model.encode(text).squeeze().tolist())

    def decode_token(self, token_id: int) -> str:
        """Return the token string for a given token ID."""
        return self._vocab.get(token_id, "")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits over vocabulary for the next token."""
        return cast(list[float],
                    self._model.get_logits_from_input_ids(input_ids))

    def get_vocab(self) -> dict[int, str]:
        """Return mapping from token ID to token string."""
        return self._vocab

    @abstractmethod
    def clean_function_name(self, token: str) -> str:
        """Strip spacing/newline artifacts for function name tokens."""
        pass

    @abstractmethod
    def clean_number_tokens(self, token: str) -> str:
        """Strip spacing/newline artifacts for number tokens."""
        pass

    @abstractmethod
    def clean_str_tokens(self, token: str) -> str:
        """Normalize spacing artifacts for string tokens."""
        pass


class Qwen3LLM(BaseLLM):
    """LLM wrapper for Qwen/Qwen3-0.6B via the llm_sdk package."""

    def __init__(self, device: str = "cpu") -> None:
        """Initialize Qwen3-0.6B, load weights and vocabulary.

        Args:
            device: The compute device to use, either 'cpu', 'cuda', or 'mps'.
        """
        print_llm_initializer("Qwen/Qwen3-0.6B")
        super().__init__(model_name="Qwen/Qwen3-0.6B", device=device)

    def clean_function_name(self, token: str) -> str:
        """Strip Ġ, ĉ, Ċ artifacts and whitespace for function name tokens."""
        return token.replace("Ġ", "").replace("ĉ", "").replace("Ċ", "").strip()

    def clean_number_tokens(self, token: str) -> str:
        """Strip Ġ, ĉ, Ċ artifacts for numeric tokens."""
        return token.replace("Ġ", "").replace("ĉ", "").replace("Ċ", "")

    def clean_str_tokens(self, token: str) -> str:
        """Replace Ġ with space and strip ĉ, Ċ artifacts for string tokens."""
        return token.replace("Ġ", " ").replace("ĉ", "").replace("Ċ", "")


class Qwen2LLM(BaseLLM):
    """LLM wrapper for Qwen/Qwen2-0.5B via the llm_sdk package.

    Uses different token cleaning than Qwen3 due to different
    tokenizer special characters (▁ instead of ĉ/Ċ).
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize Qwen2-0.5B, load weights and vocabulary.

        Args:
            device: The compute device to use, either 'cpu', 'cuda', or 'mps'.
        """
        print_llm_initializer("Qwen/Qwen2-0.5B")
        super().__init__(model_name="Qwen/Qwen2-0.5B", device=device)

    def clean_function_name(self, token: str) -> str:
        """Strip Ġ, newline artifacts and whitespace for function name tokens.
        """
        return token.replace("Ġ", "").replace("\n", "").strip()

    def clean_number_tokens(self, token: str) -> str:
        """Strip Ġ and ▁ artifacts for numeric tokens."""
        return token.replace("Ġ", "").replace("▁", "").strip()

    def clean_str_tokens(self, token: str) -> str:
        """Replace Ġ and ▁ with space and strip newline artifacts for string
        tokens.
        """
        return token.replace("Ġ", " ").replace("▁", " ").replace("\n", "")

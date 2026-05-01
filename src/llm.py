from abc import ABC, abstractmethod
from llm_sdk import Small_LLM_Model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from .shell_prints import print_llm_initializer


class BaseLLM(ABC):
    """Abstract base class for LLM wrappers."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        pass

    @abstractmethod
    def decode_token(self, token_id: int) -> str:
        pass

    @abstractmethod
    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits over vocabulary for the next token."""
        pass

    @abstractmethod
    def get_vocab(self) -> dict[int, str]:
        """Return mapping from token ID to token string."""
        pass

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
    def __init__(self, device: str = "cpu") -> None:
        print_llm_initializer("Qwen/Qwen3-0.6B")
        self._model = Small_LLM_Model(device=device)
        self._vocab = self._load_vocab()
        self.cache = {}

    def _load_vocab(self) -> dict[int, str]:
        vocab_path = self._model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            token_to_id: dict[str, int] = json.load(f)
        return {v: k for k, v in token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        return self._model.encode(text).squeeze().tolist()

    def decode_token(self, token_id: int) -> str:
        return self._vocab.get(token_id, "")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        return self._model.get_logits_from_input_ids(input_ids)

    def get_vocab(self) -> dict[int, str]:
        return self._vocab

    def clean_function_name(self, token: str) -> str:
        return token.replace("Ġ", "").replace("ĉ", "").replace("Ċ", "").strip()

    def clean_number_tokens(self, token: str) -> str:
        return token.replace("Ġ", "").replace("ĉ", "").replace("Ċ", "")

    def clean_str_tokens(self, token: str) -> str:
        return token.replace("Ġ", " ").replace("ĉ", "").replace("Ċ", "")


class Qwen2LLM(BaseLLM):

    def __init__(self) -> None:
        print_llm_initializer("Qwen/Qwen2-0.5B")
        self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self._model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-0.5B",
            torch_dtype=torch.float32,
        )
        self._model.eval()
        vocab = self._tokenizer.get_vocab()
        self._id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: str):
        return self._tokenizer.decode([token_id])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            outputs = self._model(input_tensor)
        return outputs.logits[0, -1].tolist()

    def get_vocab(self) -> dict[int, str]:
        return self._id_to_token

    def clean_function_name(self, token: str) -> str:
        return token.replace("Ġ", "").replace("\n", "").strip()

    def clean_number_tokens(self, token: str) -> str:
        return token.replace("Ġ", "").replace("▁", "").strip()

    def clean_str_tokens(self, token: str) -> str:
        return token.replace("Ġ", " ").replace("▁", " ").replace("\n", "")

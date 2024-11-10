# ==================================================================================================================
# Copyright 2024 Luca Della Libera.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# ==================================================================================================================

"""LLaMA 3 inference test (text completion).

Requires downloading the model weights and tokenizer (pretrained variant, e.g. Llama3.2-1B). Check the official
website for instructions on how to download the models (https://github.com/meta-llama/llama3#download). It is
recommended to run this script on a machine with at least 1 GPU.

NOTE: when using fp16 for generating long sequences (w/o sampling), the outputs of jitted vs non-jitted as well as
      KV-cached vs non-KV-cached tend to differ in the rightmost tokens (most likely due to error propagation).

"""

# Adapted from:
# https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/tokenizer.py
# https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/generation.py

try:
    import tiktoken
except ImportError:
    raise ImportError("`pip install tiktoken` to run this script")

import argparse
import json
import os
import time
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    Union,
    cast,
)

import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe

from llama3 import LlamaDecoder


logger = getLogger(__name__)


class Tokenizer:
    """Tokenizing and encoding/decoding text using the Tiktoken tokenizer."""

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


def build(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 1024,
    seed: int = 1,
) -> Dict:
    """
    Build a Llama instance by initializing and loading a model checkpoint.

    Args:
        ckpt_dir (str): Path to the directory containing checkpoint files.
        tokenizer_path (str): Path to the tokenizer file.
        max_seq_len (int): Maximum sequence length for input text.

    Returns:
        Llama: An instance of the Llama class with the loaded model and tokenizer.

    Raises:
        AssertionError: If there are no checkpoint files in the specified directory,
            or if the model parallel size does not match the number of checkpoint files.

    """
    assert (
        1 <= max_seq_len <= 8192
    ), f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
    assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
    assert os.path.isfile(
        tokenizer_path
    ), f"Tokenizer file '{tokenizer_path}' does not exist."

    # seed must be the same in all processes
    torch.manual_seed(seed)

    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    ckpt_path = checkpoints[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = params
    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert model_args["vocab_size"] == tokenizer.n_words
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # Uncomment to use fp16
        # if torch.cuda.is_bf16_supported():
        #    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        # else:
        #    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_args["max_seq_len"] = max_seq_len
    model = LlamaDecoder(**model_args)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return {"model": model, "tokenizer": tokenizer}


# Test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA 3 inference test")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join(
            os.path.expanduser("~"), ".llama", "checkpoints", "Llama3.2-1B"
        ),
        help="Path to the LLaMA 3 checkpoint directory",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    print(f"Checkpoint path: {checkpoint_path}")

    llama = build(checkpoint_path, os.path.join(checkpoint_path, "tokenizer.model"))
    model = llama["model"]
    tokenizer = llama["tokenizer"]

    prompts = [
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that",
    ]
    bos_toks = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    max_len = max(len(x) for x in bos_toks)
    bos_toks = torch.tensor(
        [[tokenizer.bos_id] * (max_len - len(x)) + x for x in bos_toks]
    )
    ts = time.time()
    # Optionally JIT the model
    # model = model.jit()
    hyps = model.generate(
        bos_toks,
        eos_id=tokenizer.eos_id,
        max_gen_toks=100,
        top_p=0.9,
        temp=0.6,
        use_kv_cache=True,
    )
    torch.cuda.synchronize()
    print(f"Generated in {time.time() - ts:.2f} seconds")
    hyps = [tokenizer.decode(x.tolist()) for x in hyps]
    for prompt, hyp in zip(prompts, hyps):
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Continuation: {hyp}")
    print("=" * 80)

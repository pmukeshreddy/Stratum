"""Practical model-aware estimates; provider usage is authoritative, never fabricated."""

import math
from functools import lru_cache

from .storage import encode


@lru_cache(maxsize=16)
def encoder(model):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        if model.startswith(("gpt-", "o1", "o3", "o4", "codex")):
            return tiktoken.get_encoding("o200k_base")
        return None


def estimate(value, model=None):
    text = encode(value)
    codec = encoder(model) if model else None
    if codec:
        return math.ceil(len(codec.encode(text, disallowed_special=())) * 1.12) + 64
    # Unknown model/tokenizer: a conservative ceiling instead of assuming English.
    return len(text.encode("utf-8")) + 64


def method(model):
    codec = encoder(model)
    return {
        "method": codec.name if codec else "utf8_conservative_fallback",
        "margin": 1.12 if codec else 1,
        "authoritative": False,
    }

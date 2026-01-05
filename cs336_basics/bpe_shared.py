"""Shared BPE regex utilities used by training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import regex as re

if TYPE_CHECKING:
    from collections.abc import Sequence

# Tokenisation pattern borrowed from GPT-2.
PATTERN: Final[str] = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Module-level caches to avoid recompiling regex patterns.
COMPILED_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {}
COMPILED_SPECIAL_SPLIT_PATTERNS: dict[tuple[str, tuple[str, ...]], re.Pattern[str]] = {}


def get_or_compile_pattern(pattern: str) -> re.Pattern[str]:
    """Return a cached regex pattern for the provided pattern string."""
    return COMPILED_TOKEN_PATTERNS.setdefault(pattern, re.compile(pattern))


def get_or_compile_special_split_pattern(
    special_tokens: Sequence[str],
) -> re.Pattern[str] | None:
    """Return a cached regex that isolates special tokens during pre-tokenisation."""
    if not special_tokens:
        return None

    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    joined = "|".join(re.escape(token) for token in sorted_tokens)
    # Use capturing group to preserve special tokens when splitting
    pattern = f"({joined})"
    key = (joined, tuple(special_tokens))
    return COMPILED_SPECIAL_SPLIT_PATTERNS.setdefault(key, re.compile(pattern))

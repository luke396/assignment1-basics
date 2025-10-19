"""Utility functions for training a Byte Pair Encoding (BPE) tokenizer."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable, Sequence
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO

import regex as re
from utility import load_bpe_msgpack, save_bpe_msgpack

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENDOFTEXT: str = "<|endoftext|>"
ENDOFTEXT_BYTES: bytes = ENDOFTEXT.encode("utf-8")

_COMPILED_PATTERNS: dict[str, re.Pattern] = {}
_COMPILED_SPECIAL_SPLITS: dict[tuple[str, tuple[str, ...]], re.Pattern] = {}


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Return sorted byte offsets for splitting a file into processing chunks.

    Each chunk boundary is aligned with the first occurrence of the provided
    `split_special_token` after the initial evenly spaced guess.  This ensures
    that chunks can be processed independently without slicing a special token.
    The first boundary is always ``0`` and the last boundary is the file size.
    """
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be provided as bytes")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = max(1, file_size // desired_num_chunks)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[0] = 0
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead in reasonably sized blocks.

    for idx in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[idx]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[idx] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[idx] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int = 270,
    special_tokens: Sequence[str] | None = None,
    num_workers: int | None = None,
    pattern: str = PATTERN,
    save: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a simple BPE vocabulary from the provided text file.

    Args:
        input_path: Path to the training corpus (UTF-8 encoded).
        vocab_size: Target vocabulary size including the base 256 byte tokens.
        special_tokens: Optional additional tokens reserved at the beginning of
            the vocabulary.
        num_workers: Optional worker count for multiprocessing.  Defaults to
            the CPU count, but never exceeds the number of chunks available.
        pattern: Regular expression used for pre-tokenisation.
        save: Save resulte. Defaults False.

    Returns:
        A tuple ``(vocab, merges)`` where:
            * ``vocab`` maps token indices to their byte representations.
            * ``merges`` captures the ordered sequence of byte pair merges.
    """
    resolved_special_tokens = _normalise_special_tokens(special_tokens)

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, _effective_worker_count(num_workers), ENDOFTEXT_BYTES)

    ranges: list[tuple[int, int]] = list(zip(boundaries[:-1], boundaries[1:]))
    worker_count = _effective_worker_count(num_workers, len(ranges))

    # Always use multiprocessing and have workers read their own file ranges.
    pre_token_counts = _collect_pre_token_counts_from_ranges(
        str(input_path), ranges, resolved_special_tokens, pattern, worker_count
    )

    vocab_list = _build_initial_vocab(resolved_special_tokens)
    pair_counts, token_splits = _initialise_pair_counts(pre_token_counts)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab_list) < vocab_size:
        pair_to_merge = _select_most_frequent_pair(pair_counts)
        if pair_to_merge is None:
            break
        new_token = _apply_merge(pair_to_merge, pre_token_counts, token_splits, pair_counts)
        merges.append(pair_to_merge)
        vocab_list.append(new_token)

    vocab = {idx: token for idx, token in enumerate(vocab_list)}

    if save:
        base_path = Path(input_path)
        out_path = base_path.parent / "tokenizer.msgpack.gz"
        save_bpe_msgpack(vocab, merges, out_path)

    return vocab, merges


def _normalise_special_tokens(tokens: Sequence[str] | None) -> list[str]:
    """Return a de-duplicated, ordered list of special tokens."""

    if not tokens:
        return [ENDOFTEXT]

    seen: set[str] = set()
    normalised: list[str] = []
    for token in tokens:
        if token not in seen:
            normalised.append(token)
            seen.add(token)
    return normalised


def _effective_worker_count(num_workers: int | None, num_chunks: int | None = None) -> int:
    """Pick a worker count that respects CPU availability and chunk count."""
    available = cpu_count() or 1
    requested = num_workers or available
    capped = min(requested, available)
    if num_chunks is not None:
        capped = min(capped, max(1, num_chunks))
    return max(1, capped)


def _get_compiled_pattern(pattern: str) -> re.Pattern:
    pat = _COMPILED_PATTERNS.get(pattern)
    if pat is None:
        pat = re.compile(pattern)
        _COMPILED_PATTERNS[pattern] = pat
    return pat


def _get_compiled_special_split(special_tokens: Sequence[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    key = ("|".join(re.escape(token) for token in special_tokens), tuple(special_tokens))
    pat = _COMPILED_SPECIAL_SPLITS.get(key)
    if pat is None:
        pat = re.compile(key[0])
        _COMPILED_SPECIAL_SPLITS[key] = pat
    return pat


def _collect_pre_token_counts_from_ranges(
    input_path: str,
    ranges: Sequence[tuple[int, int]],
    special_tokens: Sequence[str],
    pattern: str,
    worker_count: int,
) -> Counter[bytes]:
    """Pre-tokenise by reading file slices in workers to avoid sending large strings.

    Each task receives only offsets and small metadata, dramatically reducing
    pickling and IPC overhead compared to sending whole chunk strings.
    """
    if not ranges:
        return Counter()

    args = [(input_path, start, end, list(special_tokens), pattern) for (start, end) in ranges]
    with Pool(processes=worker_count) as pool:
        counters = pool.map(_process_range_for_pretokenization, args)

    aggregated: Counter[bytes] = Counter()
    for counter in counters:
        aggregated.update(counter)
    return aggregated


def _process_range_for_pretokenization(args: tuple[str, int, int, list[str], str]) -> Counter[bytes]:
    """Worker: open file, read [start:end], and tokenise like chunk path."""
    path, start, end, special_tokens, pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8", errors="ignore")

    pre_tokens: Counter[bytes] = Counter()
    special_split_re = _get_compiled_special_split(special_tokens)
    pat_re = _get_compiled_pattern(pattern)

    segments = special_split_re.split(data) if special_split_re else [data]
    for segment in segments:
        if not segment:
            continue
        pre_tokens.update(m.group(0).encode("utf-8") for m in pat_re.finditer(segment))
    return pre_tokens


def _build_initial_vocab(special_tokens: Iterable[str]) -> list[bytes]:
    """Construct the starting vocabulary of special tokens plus single bytes."""
    vocab_list = [token.encode("utf-8") for token in special_tokens]
    vocab_list.extend(bytes([byte]) for byte in range(256))
    return vocab_list


def _initialise_pair_counts(
    pre_token_counts: Counter[bytes],
) -> tuple[Counter[tuple[bytes, bytes]], dict[bytes, list[bytes]]]:
    """Initialise pair statistics used by the merge loop."""
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    token_splits: dict[bytes, list[bytes]] = {}

    for token_bytes, frequency in pre_token_counts.items():
        split_bytes = [bytes([byte]) for byte in token_bytes]
        token_splits[token_bytes] = split_bytes
        for idx in range(len(split_bytes) - 1):
            pair = (split_bytes[idx], split_bytes[idx + 1])
            pair_counts[pair] += frequency

    return pair_counts, token_splits


def _select_most_frequent_pair(
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes] | None:
    """Return the most frequent pair, or ``None`` if no positive counts remain."""
    positive_pairs = [(pair, count) for pair, count in pair_counts.items() if count > 0]
    if not positive_pairs:
        return None

    # Match legacy behaviour: prefer lexicographically larger pairs on ties.
    pair, _ = max(positive_pairs, key=lambda item: (item[1], item[0]))
    return pair


def _apply_merge(
    pair: tuple[bytes, bytes],
    pre_token_counts: Counter[bytes],
    token_splits: dict[bytes, list[bytes]],
    pair_counts: Counter[tuple[bytes, bytes]],
) -> bytes:
    """Apply a merge to all token splits and update neighbouring pair counts."""
    merge_a, merge_b = pair
    new_token = merge_a + merge_b

    for original_token, split_tokens in token_splits.items():
        frequency = pre_token_counts[original_token]
        if frequency == 0 or new_token not in original_token:
            continue

        idx = 0
        while idx < len(split_tokens) - 1:
            if split_tokens[idx] == merge_a and split_tokens[idx + 1] == merge_b:
                if idx > 0:
                    old_prev_pair = (split_tokens[idx - 1], split_tokens[idx])
                    pair_counts[old_prev_pair] -= frequency
                    new_prev_pair = (split_tokens[idx - 1], new_token)
                    pair_counts[new_prev_pair] += frequency
                if idx + 2 < len(split_tokens):
                    old_next_pair = (split_tokens[idx + 1], split_tokens[idx + 2])
                    pair_counts[old_next_pair] -= frequency
                    new_next_pair = (new_token, split_tokens[idx + 2])
                    pair_counts[new_next_pair] += frequency

                split_tokens[idx : idx + 2] = [new_token]
            else:
                idx += 1

    pair_counts[pair] = 0
    return new_token


if __name__ == "__main__":
    # test/debug
    # train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt")

    # train
    train_bpe(input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=1000, save=True)
    voacb, merge = load_bpe_msgpack(Path("data/tokenizer.msgpack.gz"))
    print("\nvocab:", voacb)
    print("\nmerge:", merge)

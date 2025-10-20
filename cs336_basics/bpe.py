"""Utility functions for training a Byte Pair Encoding (BPE) tokenizer."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import partial
from heapq import heapify, heappop, heappush
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO, Final, TypeAlias

import regex as re

from .utility import print_bpe_result, save_bpe_msgpack

PATTERN: Final[str] = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
ENDOFTEXT: Final[str] = "<|endoftext|>"

ChunkRange: TypeAlias = tuple[int, int]
MergePair: TypeAlias = tuple[int, int]
MergeHeapEntry: TypeAlias = tuple[int, tuple[int, ...], tuple[int, ...], int, int, int]
TokenIdSplit: TypeAlias = list[int]

_COMPILED_TOKEN_PATTERNS: dict[str, re.Pattern] = {}
_COMPILED_SPECIAL_SPLIT_PATTERNS: dict[tuple[str, tuple[str, ...]], re.Pattern] = {}
_WORKER_REGEX_CACHE: dict[str, re.Pattern | None] = {}


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

    read_ahead_bytes = 4096  # Read ahead in reasonably sized blocks.

    for idx in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[idx]
        file.seek(initial_position)
        while True:
            read_window = file.read(read_ahead_bytes)
            if read_window == b"":
                chunk_boundaries[idx] = file_size
                break

            found_at = read_window.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[idx] = initial_position + found_at
                break
            initial_position += read_ahead_bytes

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
        save: Whether to store the resulting tokenizer as ``tokenizer.msgpack.gz``.

    Returns:
        A tuple ``(vocab, merges)`` where:
            * ``vocab`` maps token indices to their byte representations.
            * ``merges`` captures the ordered sequence of byte pair merges.
    """
    resolved_special_tokens = _normalize_special_tokens(special_tokens)

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file, _effective_worker_count(num_workers), ENDOFTEXT.encode("utf-8")
        )
    chunk_ranges: list[ChunkRange] = list(zip(boundaries[:-1], boundaries[1:]))
    num_ranges = len(chunk_ranges)
    worker_count = _effective_worker_count(num_workers, num_ranges)
    pre_token_counts = _collect_pre_token_counts_from_ranges(
        str(input_path), chunk_ranges, resolved_special_tokens, pattern, worker_count
    )

    vocab_list = _build_initial_vocab(resolved_special_tokens)
    vocab_lexkey = [_lexkey(tok) for tok in vocab_list]

    pair_counts, token_splits, pair_to_tokens, pair_version = _initialize_pair_statistics(
        pre_token_counts,
        len(resolved_special_tokens),
    )
    merge_candidate_heap: list[MergeHeapEntry] = _build_merge_candidate_heap(
        pair_counts, vocab_lexkey
    )

    merged_index_pairs: list[MergePair] = []
    while len(vocab_list) < vocab_size:
        pair_to_merge: MergePair | None = _select_most_frequent_pair(
            merge_candidate_heap, pair_version
        )
        if pair_to_merge is None:
            break
        new_token_id = len(vocab_list)
        new_token = vocab_list[pair_to_merge[0]] + vocab_list[pair_to_merge[1]]
        vocab_list.append(new_token)
        vocab_lexkey.append(_lexkey(new_token))
        _apply_merge(
            new_token_id,
            pair_to_merge,
            pair_to_tokens,
            pre_token_counts,
            token_splits,
            pair_counts,
            pair_version,
            merge_candidate_heap,
            vocab_lexkey,
        )
        merged_index_pairs.append(pair_to_merge)

    vocab = {idx: token for idx, token in enumerate(vocab_list)}
    merges = [(vocab_list[i], vocab_list[j]) for i, j in merged_index_pairs]

    if save:
        base_path = Path(input_path)
        out_path = base_path.parent / "tokenizer.msgpack.gz"
        save_bpe_msgpack(vocab, merges, out_path)

    return vocab, merges


def _lexkey(token_bytes: bytes) -> tuple[int, ...]:
    """Return a lexical ordering key used for heap tie-breaking."""
    return tuple(255 - b for b in token_bytes) + (255,)


def _normalize_special_tokens(tokens: Sequence[str] | None) -> list[str]:
    """Return a de-duplicated, ordered list of special tokens."""

    if not tokens:
        return [ENDOFTEXT]

    return list(dict.fromkeys(tokens))


def _effective_worker_count(
    num_workers: int | None, num_chunks: int | None = None
) -> int:
    """Pick a worker count that respects CPU availability and chunk count."""
    available = max(1, cpu_count() - 1)
    capped = min(num_workers or available, available)
    if num_chunks is not None:
        capped = min(capped, max(1, num_chunks))
    return max(1, capped)


def _get_or_compile_pattern(pattern: str) -> re.Pattern:
    """Return a cached regex pattern for the provided pattern string."""
    return _COMPILED_TOKEN_PATTERNS.setdefault(pattern, re.compile(pattern))


def _get_or_compile_special_split_pattern(
    special_tokens: Sequence[str],
) -> re.Pattern | None:
    """Return a cached regex that isolates special tokens during pre-tokenisation."""
    if not special_tokens:
        return None
    joined = "|".join(re.escape(token) for token in special_tokens)
    key = (joined, tuple(special_tokens))
    return _COMPILED_SPECIAL_SPLIT_PATTERNS.setdefault(key, re.compile(joined))


def _collect_pre_token_counts_from_ranges(
    input_path: str,
    chunk_ranges: Sequence[ChunkRange],
    special_tokens: Sequence[str],
    pattern: str,
    worker_count: int,
    chunksize: int | None = None,
) -> Counter[bytes]:
    """Pre-tokenise designated byte ranges in worker processes.

    Each task receives only offsets and small metadata, dramatically reducing
    pickling and IPC overhead compared to sending whole chunk strings.
    """
    if not chunk_ranges:
        return Counter()

    num_ranges = len(chunk_ranges)
    worker_func = partial(_process_range_for_pretokenization, input_path)
    if chunksize is None:
        base = max(1, num_ranges // (worker_count * 8))
        chunksize = min(256, max(4, base))

    aggregated: Counter[bytes] = Counter()
    with Pool(
        processes=worker_count,
        initializer=_initialize_worker_regex_cache,
        initargs=(tuple(special_tokens), pattern),
    ) as pool:
        for counter in pool.imap_unordered(
            worker_func, chunk_ranges, chunksize=chunksize
        ):
            aggregated.update(counter)
    return aggregated


def _initialize_worker_regex_cache(special_tokens: Sequence[str], pattern: str) -> None:
    """Cache frequently reused regex objects inside worker processes."""
    special_split_re = _get_or_compile_special_split_pattern(special_tokens)
    pattern_re = _get_or_compile_pattern(pattern)
    _WORKER_REGEX_CACHE["special"] = special_split_re
    _WORKER_REGEX_CACHE["pattern"] = pattern_re


def _process_range_for_pretokenization(
    path: str, offset_span: ChunkRange
) -> Counter[bytes]:
    """Worker helper: read a byte span and emit regex-pre-tokenised counts."""
    start, end = offset_span
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8", errors="ignore")

    pre_tokens: Counter[bytes] = Counter()
    special_split_re = _WORKER_REGEX_CACHE["special"]
    pattern_re = _WORKER_REGEX_CACHE["pattern"]
    assert pattern_re is not None

    segments = special_split_re.split(data) if special_split_re else [data]
    for segment in segments:
        if not segment:
            continue
        pre_tokens.update(
            match.group(0).encode("utf-8") for match in pattern_re.finditer(segment)
        )
    return pre_tokens


def _build_initial_vocab(special_tokens: Iterable[str]) -> list[bytes]:
    """Construct the starting vocabulary of special tokens plus single bytes."""
    return [token.encode("utf-8") for token in special_tokens] + [
        bytes([byte]) for byte in range(256)
    ]


def _initialize_pair_statistics(
    pre_token_counts: Counter[bytes], symbol_offset: int
) -> tuple[
    Counter[MergePair],
    dict[bytes, TokenIdSplit],
    dict[MergePair, set[bytes]],
    dict[MergePair, int],
]:
    """Initialize pair statistics used by the merge loop."""
    pair_counts: Counter[MergePair] = Counter()
    token_splits: dict[bytes, TokenIdSplit] = {}
    pair_to_tokens: dict[MergePair, set[bytes]] = defaultdict(set)
    pair_version: dict[MergePair, int] = defaultdict(int)

    for token_bytes, frequency in pre_token_counts.items():
        split_int: TokenIdSplit = [b + symbol_offset for b in token_bytes]
        token_splits[token_bytes] = split_int
        for idx in range(len(split_int) - 1):
            pair: MergePair = (split_int[idx], split_int[idx + 1])
            pair_counts[pair] += frequency
            pair_to_tokens[pair].add(token_bytes)

    return pair_counts, token_splits, pair_to_tokens, pair_version


def _build_merge_candidate_heap(
    pair_counts: Counter[MergePair],
    vocab_lexkey: Sequence[tuple[int, ...]],
) -> list[MergeHeapEntry]:
    """Populate a heap of merge candidates ordered by frequency and lexical key."""
    merge_candidate_heap = [
        (-freq, vocab_lexkey[a], vocab_lexkey[b], a, b, 0)
        for (a, b), freq in pair_counts.items()
        if freq > 0
    ]
    heapify(merge_candidate_heap)
    return merge_candidate_heap


def _select_most_frequent_pair(
    merge_candidate_heap: list[MergeHeapEntry],
    pair_version: dict[MergePair, int],
) -> MergePair | None:
    """Return the most frequent pair, or ``None`` if no positive counts remain."""
    while merge_candidate_heap:
        neg_count, _, _, a, b, version = heappop(merge_candidate_heap)
        if neg_count < 0 and pair_version[a, b] == version:
            return a, b
    return None


def _apply_merge(
    new_token_id: int,
    pair: MergePair,
    pair_to_tokens: dict[MergePair, set[bytes]],
    pre_token_counts: Counter[bytes],
    token_splits: dict[bytes, TokenIdSplit],
    pair_counts: Counter[MergePair],
    pair_version: dict[MergePair, int],
    merge_candidate_heap: list[MergeHeapEntry],
    vocab_lexkey: Sequence[tuple[int, ...]],
):
    """Merge the selected pair across all tokens and refresh adjacent pair counts."""
    tokens = pair_to_tokens[pair]
    a, b = pair
    pair_count_delta: Counter[MergePair] = Counter()
    for token in list(tokens):
        split: TokenIdSplit = token_splits[token]
        freq = pre_token_counts[token]
        new_split: TokenIdSplit = []
        i = 0
        n = len(split)
        while i < n:
            if i + 1 < n and split[i] == a and split[i + 1] == b:
                prev = split[i - 1] if i > 0 else None
                nxt = split[i + 2] if i + 2 < n else None
                if prev is not None:
                    pair_count_delta[(prev, a)] -= freq
                    pair_count_delta[(prev, new_token_id)] += freq
                if nxt is not None:
                    pair_count_delta[(b, nxt)] -= freq
                    pair_count_delta[(new_token_id, nxt)] += freq
                new_split.append(new_token_id)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        old_pairs = {(split[i], split[i + 1]) for i in range(len(split) - 1)}
        new_pairs = {(new_split[j], new_split[j + 1]) for j in range(len(new_split) - 1)}
        for p in new_pairs - old_pairs:
            pair_to_tokens[p].add(token)
        for p in old_pairs - new_pairs:
            pair_to_tokens[p].discard(token)
        token_splits[token] = new_split

    pair_counts.update(pair_count_delta)
    pair_counts[pair] = 0

    for a, b in pair_count_delta:
        freq = pair_counts[a, b]
        pair_version[a, b] += 1  # freq <= 0, ignore and not pushed
        if freq > 0:
            heappush(
                merge_candidate_heap,
                (-freq, vocab_lexkey[a], vocab_lexkey[b], a, b, pair_version[a, b]),
            )
    pair_version[pair] += 1


if __name__ == "__main__":
    # test/debug
    # train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt")

    # train
    bpe_result = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, save=False
    )
    print_bpe_result(bpe_result=bpe_result)

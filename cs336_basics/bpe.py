"""Utility functions for training a Byte Pair Encoding (BPE) tokenizer."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import partial
from heapq import heapify, heappop, heappush
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO, Final, TypeAlias

import regex as re

from .utility import print_bpe_result

PATTERN: Final[str] = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
ENDOFTEXT: Final[str] = "<|endoftext|>"

ChunkRange: TypeAlias = tuple[int, int]
MergePair: TypeAlias = tuple[int, int]
MergeHeapEntry: TypeAlias = tuple[int, tuple[int, ...], tuple[int, ...], int, int, int]
TokenIdSplit: TypeAlias = list[int]

COMPILED_TOKEN_PATTERNS: dict[str, re.Pattern] = {}
COMPILED_SPECIAL_SPLIT_PATTERNS: dict[tuple[str, tuple[str, ...]], re.Pattern] = {}
WORKER_REGEX_CACHE: dict[str, re.Pattern | None] = {}

logger = logging.getLogger(__name__)


def _setup_logging(log_path: Path | str | None = None) -> None:
    """Configure logger to write to stdout (and optionally a file)."""
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path:
        log_path_obj = Path(log_path)
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _artifact_paths(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    output_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Return the vocab/merge artifact file paths for a dataset."""
    base_path = Path(input_path)
    dataset_name = base_path.stem
    out_directory = Path(output_dir) if output_dir else base_path.parent
    vocab_out = out_directory / f"tokenizer_vocab_{dataset_name}_{vocab_size}.json"
    merges_out = out_directory / f"tokenizer_merges_{dataset_name}_{vocab_size}.json"
    return vocab_out, merges_out


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
    output_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a simple BPE vocabulary from the provided text file.

    Note:
        This function expects logging to be configured by the caller if logging
        output is desired.

    Args:
        input_path: Path to the training corpus (UTF-8 encoded).
        vocab_size: Target vocabulary size including the base 256 byte tokens.
        special_tokens: Optional additional tokens reserved at the beginning of
            the vocabulary.
        num_workers: Optional worker count for multiprocessing.  Defaults to
            the CPU count, but never exceeds the number of chunks available.
        pattern: Regular expression used for pre-tokenisation.
        save: Whether to save the tokenizer artifacts.
        output_dir: Directory where tokenizer artifacts will be saved.
            Defaults to the same directory as the input file.

    Returns:
        A tuple ``(vocab, merges)`` where:
            * ``vocab`` maps token indices to their byte representations.
            * ``merges`` captures the ordered sequence of byte pair merges.
    """
    resolved_special_tokens = _normalize_special_tokens(special_tokens)

    requested_chunks = _effective_worker_count(num_workers)
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file, requested_chunks, ENDOFTEXT.encode("utf-8")
        )
    file_size = boundaries[-1] if boundaries else 0
    chunk_ranges: list[ChunkRange] = list(zip(boundaries[:-1], boundaries[1:]))
    num_ranges = len(chunk_ranges)
    worker_count = _effective_worker_count(num_workers, num_ranges)
    logger.info(
        "Chunking complete for %s: file_size=%d bytes, desired_chunks=%d, actual_chunks=%d, worker_count=%d",
        input_path,
        file_size,
        requested_chunks,
        num_ranges,
        worker_count,
    )
    pre_token_counts = _collect_pre_token_counts_from_ranges(
        str(input_path), chunk_ranges, resolved_special_tokens, pattern, worker_count
    )
    total_pre_tokens = sum(pre_token_counts.values())
    logger.info(
        "Collected pre-token counts: unique_tokens=%d, total_tokens=%d",
        len(pre_token_counts),
        total_pre_tokens,
    )

    vocab_list = _build_initial_vocab(resolved_special_tokens)
    vocab_lexkey = [_lexkey(tok) for tok in vocab_list]
    logger.debug(
        "Initial vocab constructed: size=%d (includes %d special tokens)",
        len(vocab_list),
        len(resolved_special_tokens),
    )

    pair_counts, token_splits, pair_to_tokens, pair_version = _initialize_pair_statistics(
        pre_token_counts,
        len(resolved_special_tokens),
    )
    logger.info(
        "Pair statistics initialized: pair_counts=%d, tracked_tokens=%d",
        len(pair_counts),
        len(token_splits),
    )
    merge_candidate_heap: list[MergeHeapEntry] = _build_merge_candidate_heap(
        pair_counts, vocab_lexkey
    )

    merged_index_pairs: list[MergePair] = []
    try:
        while len(vocab_list) < vocab_size:
            pair_to_merge: MergePair | None = _select_most_frequent_pair(
                merge_candidate_heap, pair_version
            )
            if pair_to_merge is None:
                logger.warning(
                    "Merge loop ended early: no more pairs to merge after %d merges",
                    len(merged_index_pairs),
                )
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
    except MemoryError:
        logger.warning(
            "MemoryError during merge loop: completed_merges=%d, current_vocab_size=%d",
            len(merged_index_pairs),
            len(vocab_list),
        )
        raise

    logger.info(
        "Merge loop finished: total_merges=%d, final_vocab_size=%d (target=%d)",
        len(merged_index_pairs),
        len(vocab_list),
        vocab_size,
    )

    vocab = {idx: token for idx, token in enumerate(vocab_list)}
    merges = [(vocab_list[i], vocab_list[j]) for i, j in merged_index_pairs]

    if save:
        vocab_out, merges_out = _artifact_paths(input_path, vocab_size, output_dir)

        vocab_payload = _format_vocab(vocab)
        merges_payload = _format_merges(merges)

        vocab_out.parent.mkdir(parents=True, exist_ok=True)
        merges_out.parent.mkdir(parents=True, exist_ok=True)

        with vocab_out.open("w", encoding="utf-8") as vocab_file:
            json.dump(vocab_payload, vocab_file, indent=2, ensure_ascii=True)
            vocab_file.write("\n")

        with merges_out.open("w", encoding="utf-8") as merges_file:
            json.dump(merges_payload, merges_file, indent=2, ensure_ascii=True)
            merges_file.write("\n")

        try:
            vocab_size_bytes = vocab_out.stat().st_size
            merges_size_bytes = merges_out.stat().st_size
        except OSError:
            vocab_size_bytes = merges_size_bytes = -1
        logger.info(
            "Tokenizer saved to %s (size=%d bytes) and %s (size=%d bytes)",
            vocab_out,
            vocab_size_bytes,
            merges_out,
            merges_size_bytes,
        )

    return vocab, merges


def _format_vocab(vocab: dict[int, bytes]) -> dict[str, str]:
    """Convert vocab mapping into a JSON-serialisable dict."""
    return {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()}


def _format_merges(merges: list[tuple[bytes, bytes]]) -> list[list[str]]:
    """Convert merges sequence into a JSON-serialisable list."""
    return [[left.hex(), right.hex()] for left, right in merges]


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


def get_or_compile_pattern(pattern: str) -> re.Pattern:
    """Return a cached regex pattern for the provided pattern string."""
    return COMPILED_TOKEN_PATTERNS.setdefault(pattern, re.compile(pattern))


def get_or_compile_special_split_pattern(
    special_tokens: Sequence[str],
) -> re.Pattern | None:
    """Return a cached regex that isolates special tokens during pre-tokenisation."""
    assert special_tokens is not None
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    joined = "|".join(re.escape(token) for token in sorted_tokens)
    # Use capturing group to preserve special tokens when splitting
    pattern = f"({joined})"
    key = (joined, tuple(special_tokens))
    return COMPILED_SPECIAL_SPLIT_PATTERNS.setdefault(key, re.compile(pattern))


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
        if worker_count >= num_ranges:
            chunksize = 1
        else:
            base = max(1, num_ranges // (worker_count * 8))
            chunksize = min(256, base)

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
    special_split_re = get_or_compile_special_split_pattern(special_tokens)
    pattern_re = get_or_compile_pattern(pattern)
    WORKER_REGEX_CACHE["special"] = special_split_re
    WORKER_REGEX_CACHE["pattern"] = pattern_re


def _process_range_for_pretokenization(
    path: str, offset_span: ChunkRange
) -> Counter[bytes]:
    """Worker helper: read a byte span and emit regex-pre-tokenised counts."""
    start, end = offset_span
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8", errors="ignore")

    pre_tokens: Counter[bytes] = Counter()
    special_split_re = WORKER_REGEX_CACHE["special"]
    pattern_re = WORKER_REGEX_CACHE["pattern"]
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
    import argparse

    DATASET_CONFIGS = {
        "tinystories_valid": {
            "path": "data/TinyStoriesV2-GPT4-valid.txt",
            "default_vocab_size": 1000,
        },
        "tinystories_train": {
            "path": "data/TinyStoriesV2-GPT4-train.txt",
            "default_vocab_size": 10000,
        },
        "owt_valid": {
            "path": "data/owt_valid.txt",
            "default_vocab_size": 1000,
        },
        "owt_train": {
            "path": "data/owt_train.txt",
            "default_vocab_size": 32000,
        },
    }

    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to training corpus (takes precedence over --dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Predefined dataset name",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size (required if using --input-path)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for multiprocessing (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for saving tokenizer artifacts (default: same as input)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving tokenizer files",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: stdout only)",
    )

    args = parser.parse_args()

    if args.input_path:
        input_path = args.input_path
        if not args.vocab_size:
            parser.error("--vocab-size is required when using --input-path")
        vocab_size = args.vocab_size
    elif args.dataset:
        config = DATASET_CONFIGS[args.dataset]
        input_path = config["path"]
        vocab_size = args.vocab_size or config["default_vocab_size"]
    else:
        parser.error("Either --input-path or --dataset must be specified")

    _setup_logging(args.log_file)
    logger.info(
        "Training BPE: input=%s, vocab_size=%d, workers=%s",
        input_path,
        vocab_size,
        args.num_workers or "auto",
    )

    bpe_result = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        num_workers=args.num_workers,
        save=not args.no_save,
        output_dir=args.output_dir,
    )

    print_bpe_result(bpe_result=bpe_result)

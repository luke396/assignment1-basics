"""Utilities for experimenting with TinyStories and OWT tokenizers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from cs336_basics.tokenizer import Tokenizer

if TYPE_CHECKING:
    from collections.abc import Generator


BYTES_PER_MEGABYTE: Final[int] = 1024 * 1024
BYTES_PER_GIGABYTE: Final[int] = 1024 * 1024 * 1024
PROGRESS_UPDATE_INTERVAL: Final[float] = 0.5


@dataclass(frozen=True)
class DatasetMetrics:
    """Aggregated dataset size statistics for reporting."""

    file_size_bytes: int
    utf_len: int
    char_len: int
    utf8_to_filesize_ratio: float
    avg_bytes_per_char: float


@dataclass(frozen=True)
class TokenizerStats:
    """Runtime and cache statistics for a tokenizer benchmark run."""

    elapsed: float
    speed: float
    hits: int
    misses: int
    currsize: int
    maxsize: int


def qab(ts_tokenizer: Tokenizer, owt_tokenizer: Tokenizer) -> None:
    """Compare tokenizer compression ratios on representative samples."""
    ts_valid_sample = "u don't have to be scared of the loud dog"
    owt_valid_sample = (
        "LOUISVILLE, Ky. — A few unflattering reviews are to be expected with any "
        "hotel,"
    )

    ts_sample_len = len(ts_valid_sample.encode("utf-8"))
    owt_sample_len = len(owt_valid_sample.encode("utf-8"))

    ts_ts_tokenizer = ts_tokenizer.encode(ts_valid_sample)
    ts_owt_tokenizer = owt_tokenizer.encode(ts_valid_sample)

    owt_ts_tokenizer = ts_tokenizer.encode(owt_valid_sample)
    owt_owt_tokenizer = owt_tokenizer.encode(owt_valid_sample)

    print(
        "TS sample with TS tokenizer: ",
        ts_ts_tokenizer,
        "\ntokenizer's compression ratio:",
        f"{ts_sample_len / len(ts_ts_tokenizer):.2f}",
    )
    print(
        "TS sample with OWT tokenizer: ",
        ts_owt_tokenizer,
        "\ntokenizer's compression ratio:",
        f"{ts_sample_len / len(ts_owt_tokenizer):.2f}",
    )
    print("=" * 60)
    print(
        "OWT sample with OWT tokenizer: ",
        owt_owt_tokenizer,
        "\ntokenizer's compression ratio:",
        f"{owt_sample_len / len(owt_owt_tokenizer):.2f}",
    )
    print(
        "OWT sample with TS tokenizer: ",
        ts_owt_tokenizer,
        "\ntokenizer's compression ratio:",
        f"{owt_sample_len / len(owt_ts_tokenizer):.2f}",
    )


def qc(ts_tokenizer: Tokenizer, owt_tokenizer: Tokenizer) -> None:
    """Analyze dataset size characteristics and tokenizer performance."""
    valid_path = Path("data/TinyStoriesV2-GPT4-valid.txt")
    valid_data = valid_path.read_text(encoding="utf-8")
    metrics = _calculate_dataset_metrics(valid_path, valid_data)

    pile_size_gb = 825
    _print_file_size_analysis(metrics, pile_size_gb)
    _print_tokenizer_performance(
        ts_tokenizer,
        owt_tokenizer,
        valid_data,
        metrics,
        pile_size_gb,
    )


def _calculate_dataset_metrics(valid_path: Path, valid_data: str) -> DatasetMetrics:
    valid_data_utf_len = len(valid_data.encode("utf-8"))
    char_len = len(valid_data)
    file_size_bytes = valid_path.stat().st_size
    utf8_to_filesize_ratio = (
        valid_data_utf_len / file_size_bytes if file_size_bytes else 0.0
    )
    avg_bytes_per_char = valid_data_utf_len / char_len if char_len else 0.0
    return DatasetMetrics(
        file_size_bytes=file_size_bytes,
        utf_len=valid_data_utf_len,
        char_len=char_len,
        utf8_to_filesize_ratio=utf8_to_filesize_ratio,
        avg_bytes_per_char=avg_bytes_per_char,
    )


def _print_file_size_analysis(
    metrics: DatasetMetrics,
    pile_size_gb: int,
) -> None:
    print("=" * 60)
    print("File Size Analysis:")
    print(
        "  File size on disk: "
        f"{metrics.file_size_bytes:,} bytes "
        f"({metrics.file_size_bytes / BYTES_PER_MEGABYTE:.2f} MB)",
    )
    print(
        "  UTF-8 encoded size: "
        f"{metrics.utf_len:,} bytes "
        f"({metrics.utf_len / BYTES_PER_MEGABYTE:.2f} MB)",
    )
    print(f"  Character count: {metrics.char_len:,} characters")
    print(f"  UTF-8 / File size ratio: {metrics.utf8_to_filesize_ratio:.4f}")
    print(f"  Average bytes per character: {metrics.avg_bytes_per_char:.4f}")
    print()
    print(f"Estimation for {pile_size_gb} GB text file:")
    print(f"  If file size = {pile_size_gb} GB on disk")
    print(
        "  Estimated UTF-8 bytes = "
        f"{pile_size_gb} GB x {metrics.utf8_to_filesize_ratio:.4f}",
    )
    estimated_utf8_gb = pile_size_gb * metrics.utf8_to_filesize_ratio
    print(f"                        = {estimated_utf8_gb:.2f} GB")
    pile_size_bytes = estimated_utf8_gb * BYTES_PER_GIGABYTE
    print(f"                        = {pile_size_bytes:,.0f} bytes")


def _print_tokenizer_performance(
    ts_tokenizer: Tokenizer,
    owt_tokenizer: Tokenizer,
    valid_data: str,
    metrics: DatasetMetrics,
    pile_size_gb: int,
) -> None:
    print("=" * 60)
    print("Tokenization Performance:")
    print(f"  Validation data: {metrics.utf_len:,} bytes\n")

    pile_size_bytes = metrics.utf8_to_filesize_ratio * pile_size_gb * BYTES_PER_GIGABYTE
    ts_stats = _benchmark_tokenizer(ts_tokenizer, valid_data, metrics.utf_len)
    owt_stats = _benchmark_tokenizer(owt_tokenizer, valid_data, metrics.utf_len)

    _print_tokenizer_stats(
        "TinyStories Tokenizer",
        ts_stats,
        pile_size_bytes,
        pile_size_gb,
    )
    print()
    _print_tokenizer_stats(
        "OWT Tokenizer",
        owt_stats,
        pile_size_bytes,
        pile_size_gb,
    )


def _benchmark_tokenizer(
    tokenizer: Tokenizer,
    valid_data: str,
    data_utf_len: int,
) -> TokenizerStats:
    info_before = tokenizer._bpe_encode_cached.cache_info()  # noqa: SLF001
    start = time.time()
    _ = list(tokenizer.encode_iterable(valid_data))
    elapsed = time.time() - start
    info_after = tokenizer._bpe_encode_cached.cache_info()  # noqa: SLF001

    speed = data_utf_len / elapsed if elapsed else 0.0
    hits = info_after.hits - info_before.hits
    misses = info_after.misses - info_before.misses
    return TokenizerStats(
        elapsed=elapsed,
        speed=speed,
        hits=hits,
        misses=misses,
        currsize=info_after.currsize,
        maxsize=info_after.maxsize,  # pyright: ignore[reportArgumentType]
    )


def _print_tokenizer_stats(
    name: str,
    stats: TokenizerStats,
    pile_size_bytes: float,
    pile_size_gb: int,
) -> None:
    pile_time_cost = pile_size_bytes / stats.speed if stats.speed else float("inf")
    total_requests = stats.hits + stats.misses
    hit_rate = (stats.hits / total_requests * 100) if total_requests else 0.0

    print(f"  {name}:")
    print(f"    Time: {stats.elapsed:.2f}s  |  Speed: {stats.speed:,.2f} bytes/s")
    print(
        "    Cache: "
        f"{stats.hits:,} hits, {stats.misses:,} misses "
        f"(hit rate: {hit_rate:.2f}%)",
    )
    print(f"    Cache size: {stats.currsize:,} / {stats.maxsize:,}")
    print(
        "    Est. for "
        f"{pile_size_gb}GB: {pile_time_cost:,.2f}s "
        f"({pile_time_cost / 3600:.2f}h)",
    )


def qd(ts_tokenizer: Tokenizer, owt_tokenizer: Tokenizer) -> None:
    """Encode datasets with both tokenizers and persist token IDs to disk."""
    _encode(
        ts_tokenizer,
        "data/TinyStoriesV2-GPT4-train.txt",
        "data/ts_train_token_ids.npy",
    )
    _encode(
        ts_tokenizer,
        "data/TinyStoriesV2-GPT4-valid.txt",
        "data/ts_valid_token_ids.npy",
    )
    _encode(owt_tokenizer, "data/owt_valid.txt", "data/owt_valid_token_ids.npy")
    _encode(owt_tokenizer, "data/owt_train.txt", "data/owt_train_token_ids.npy")


def _encode(
    tokenizer: Tokenizer,
    data_path: str | Path,
    save_path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> np.ndarray:
    """Memory-efficient encoding using streaming approach.

    Reads file in chunks and processes incrementally to minimize memory usage.
    Uses encode_iterable with generator pattern to avoid loading entire file.

    Args:
        tokenizer: Tokenizer instance
        data_path: Path to input text file
        save_path: Path to save output .npy file
        chunk_size: Size of each read chunk in bytes (default: 1MB)

    """
    input_path = Path(data_path)
    output_path = Path(save_path)
    file_size = input_path.stat().st_size
    print(f"Processing {input_path} ({file_size / BYTES_PER_MEGABYTE:.2f} MB)...")
    print(f"  Chunk size: {chunk_size / 1024:.2f} KB")
    start_time = time.time()

    batch_size = 100_000  # Accumulate tokens in batches
    token_batches = []
    current_batch = []
    bytes_processed = 0

    def text_chunks() -> Generator[str, None, None]:
        """Yield text chunks from file and track progress."""
        nonlocal bytes_processed
        with input_path.open(encoding="utf-8") as infile:
            while True:
                chunk = infile.read(chunk_size)
                if not chunk:
                    break
                bytes_processed += len(chunk.encode("utf-8"))
                yield chunk

    # Process file in streaming fashion with progress tracking
    last_update_time = start_time
    for token_id in tokenizer.encode_iterable(text_chunks()):
        current_batch.append(token_id)

        # When batch is full, convert to numpy and store
        if len(current_batch) >= batch_size:
            token_batches.append(np.array(current_batch, dtype=np.uint16))
            current_batch = []

            # Update progress (throttle to every 0.5 seconds)
            current_time = time.time()
            if current_time - last_update_time >= PROGRESS_UPDATE_INTERVAL:
                progress = min(100.0, bytes_processed / file_size * 100)
                elapsed = current_time - start_time
                speed = bytes_processed / elapsed / BYTES_PER_MEGABYTE
                num_tokens = sum(len(b) for b in token_batches)
                print(
                    "\r  Progress: "
                    f"{progress:.1f}% | {speed:.2f} MB/s | {num_tokens:,} tokens",
                    end="",
                    flush=True,
                )
                last_update_time = current_time

    # Clear progress line
    print("\r" + " " * 80 + "\r", end="", flush=True)

    # Don't forget the last batch
    if current_batch:
        token_batches.append(np.array(current_batch, dtype=np.uint16))

    # Concatenate all batches into final array
    token_ids_array = (
        np.concatenate(token_batches)
        if token_batches
        else np.array([], dtype=np.uint16)
    )

    # Clear intermediate data
    del token_batches, current_batch

    # Save to disk
    np.save(output_path, token_ids_array)

    # Report results
    elapsed = time.time() - start_time
    speed = file_size / elapsed / BYTES_PER_MEGABYTE if elapsed else 0.0
    print(f"  Completed in {elapsed:.2f}s ({speed:.2f} MB/s)")
    print(f"  Saved {len(token_ids_array):,} tokens to {output_path}")

    return token_ids_array


if __name__ == "__main__":
    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tokenizer_vocab_TinyStoriesV2-GPT4-train_10000.json",
        merges_filepath="data/tokenizer_merges_TinyStoriesV2-GPT4-train_10000.json",
    )
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tokenizer_vocab_owt_train_32000.json",
        merges_filepath="data/tokenizer_merges_owt_train_32000.json",
    )

    qab(ts_tokenizer, owt_tokenizer)
    qc(ts_tokenizer, owt_tokenizer)
    qd(ts_tokenizer, owt_tokenizer)

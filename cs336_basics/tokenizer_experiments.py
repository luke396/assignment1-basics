import os
import time

import numpy as np


def qab(ts_tokenizer, owt_tokenizer):
    ts_valid_sample = "u don't have to be scared of the loud dog"
    owt_valid_sample = (
        "LOUISVILLE, Ky. — A few unflattering reviews are to be expected with any hotel,"
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


def qc(ts_tokenizer, owt_tokenizer):
    valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    with open(valid_path, encoding="utf-8") as f:
        valid_data = f.read()
    valid_data_utf_len = len(valid_data.encode("utf-8"))
    valid_data_str_len = len(valid_data)

    file_size_bytes = os.path.getsize(valid_path)

    utf8_to_filesize_ratio = valid_data_utf_len / file_size_bytes
    avg_bytes_per_char = valid_data_utf_len / valid_data_str_len

    print("=" * 60)
    print("File Size Analysis:")
    print(
        f"  File size on disk: {file_size_bytes:,} bytes ({file_size_bytes / 1024 / 1024:.2f} MB)"
    )
    print(
        f"  UTF-8 encoded size: {valid_data_utf_len:,} bytes ({valid_data_utf_len / 1024 / 1024:.2f} MB)"
    )
    print(f"  Character count: {valid_data_str_len:,} characters")
    print(f"  UTF-8 / File size ratio: {utf8_to_filesize_ratio:.4f}")
    print(f"  Average bytes per character: {avg_bytes_per_char:.4f}")
    print()
    print("Estimation for 825 GB text file:")
    print("  If file size = 825 GB on disk")
    print(f"  Estimated UTF-8 bytes = 825 GB × {utf8_to_filesize_ratio:.4f}")
    print(f"                        = {825 * utf8_to_filesize_ratio:.2f} GB")
    print(
        f"                        = {825 * utf8_to_filesize_ratio * 1024**3:,.0f} bytes"
    )

    print("=" * 60)
    print("Tokenization Performance:")
    print(f"  Validation data: {valid_data_utf_len:,} bytes\n")
    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * utf8_to_filesize_ratio * 1024**3

    # TinyStories Tokenizer
    ts_info_before = ts_tokenizer._bpe_encode_cached.cache_info()
    ts_start = time.time()
    _ = list(ts_tokenizer.encode_iterable(valid_data))
    ts_end = time.time()
    ts_info_after = ts_tokenizer._bpe_encode_cached.cache_info()

    ts_diff_time = ts_end - ts_start
    ts_speed = valid_data_utf_len / ts_diff_time
    pile_time_ts_cost = pile_size_bytes / ts_speed

    # Calculate cache stats for this encoding
    ts_hits = ts_info_after.hits - ts_info_before.hits
    ts_misses = ts_info_after.misses - ts_info_before.misses
    ts_total = ts_hits + ts_misses
    ts_hit_rate = ts_hits / ts_total * 100 if ts_total > 0 else 0

    print("  TinyStories Tokenizer:")
    print(f"    Time: {ts_diff_time:.2f}s  |  Speed: {ts_speed:,.2f} bytes/s")
    print(
        f"    Cache: {ts_hits:,} hits, {ts_misses:,} misses (hit rate: {ts_hit_rate:.2f}%)"
    )
    print(f"    Cache size: {ts_info_after.currsize:,} / {ts_info_after.maxsize:,}")
    print(
        f"    Est. for {pile_size_gb}GB: {pile_time_ts_cost:,.2f}s ({pile_time_ts_cost / 3600:.2f}h)\n"
    )

    # OWT Tokenizer
    owt_info_before = owt_tokenizer._bpe_encode_cached.cache_info()
    owt_start = time.time()
    _ = list(owt_tokenizer.encode_iterable(valid_data))
    owt_end = time.time()
    owt_info_after = owt_tokenizer._bpe_encode_cached.cache_info()

    owt_diff_time = owt_end - owt_start
    owt_speed = valid_data_utf_len / owt_diff_time
    pile_time_owt_cost = pile_size_bytes / owt_speed

    # Calculate cache stats for this encoding
    owt_hits = owt_info_after.hits - owt_info_before.hits
    owt_misses = owt_info_after.misses - owt_info_before.misses
    owt_total = owt_hits + owt_misses
    owt_hit_rate = owt_hits / owt_total * 100 if owt_total > 0 else 0

    print("  OWT Tokenizer:")
    print(f"    Time: {owt_diff_time:.2f}s  |  Speed: {owt_speed:,.2f} bytes/s")
    print(
        f"    Cache: {owt_hits:,} hits, {owt_misses:,} misses (hit rate: {owt_hit_rate:.2f}%)"
    )
    print(f"    Cache size: {owt_info_after.currsize:,} / {owt_info_after.maxsize:,}")
    print(
        f"    Est. for {pile_size_gb}GB: {pile_time_owt_cost:,.2f}s ({pile_time_owt_cost / 3600:.2f}h)"
    )


def qd(ts_tokenizer, owt_tokenizer):
    # _encode(
    #     ts_tokenizer, "data/TinyStoriesV2-GPT4-train.txt", "data/ts_train_token_ids.npy"
    # )
    # _encode(
    #     ts_tokenizer, "data/TinyStoriesV2-GPT4-valid.txt", "data/ts_valid_token_ids.npy"
    # )
    _encode(owt_tokenizer, "data/owt_valid.txt", "data/owt_valid_token_ids.npy")
    _encode(owt_tokenizer, "data/owt_train.txt", "data/owt_train_token_ids.npy")


def _encode(tokenizer, data_path, save_path, chunk_size: int = 1024 * 1024):
    """
    Memory-efficient encoding using streaming approach.

    Reads file in chunks and processes incrementally to minimize memory usage.
    Uses encode_iterable with generator pattern to avoid loading entire file.

    Args:
        tokenizer: Tokenizer instance
        data_path: Path to input text file
        save_path: Path to save output .npy file
        chunk_size: Size of each read chunk in bytes (default: 1MB)
    """
    file_size = os.path.getsize(data_path)
    print(f"Processing {data_path} ({file_size / 1024 / 1024:.2f} MB)...")
    print(f"  Chunk size: {chunk_size / 1024:.2f} KB")
    start_time = time.time()

    batch_size = 100_000  # Accumulate tokens in batches
    token_batches = []
    current_batch = []
    bytes_processed = 0

    def text_chunks():
        """Generator that yields text chunks from file and tracks progress."""
        nonlocal bytes_processed
        with open(data_path, encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
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
            if current_time - last_update_time >= 0.5:
                progress = min(100.0, bytes_processed / file_size * 100)
                elapsed = current_time - start_time
                speed = bytes_processed / elapsed / 1024 / 1024
                num_tokens = sum(len(b) for b in token_batches)
                print(
                    f"\r  Progress: {progress:.1f}% | {speed:.2f} MB/s | {num_tokens:,} tokens",
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
        np.concatenate(token_batches) if token_batches else np.array([], dtype=np.uint16)
    )

    # Clear intermediate data
    del token_batches, current_batch

    # Save to disk
    np.save(save_path, token_ids_array)

    # Report results
    elapsed = time.time() - start_time
    speed = file_size / elapsed / 1024 / 1024
    print(f"  Completed in {elapsed:.2f}s ({speed:.2f} MB/s)")
    print(f"  Saved {len(token_ids_array):,} tokens to {save_path}")

    return token_ids_array


if __name__ == "__main__":
    from .tokenizer import Tokenier

    ts_tokenizer = Tokenier.from_files(
        vocab_filepath="data/tokenizer_vocab_TinyStoriesV2-GPT4-train_10000.json",
        merges_filepath="data/tokenizer_merges_TinyStoriesV2-GPT4-train_10000.json",
    )
    owt_tokenizer = Tokenier.from_files(
        vocab_filepath="data/tokenizer_vocab_owt_train_32000.json",
        merges_filepath="data/tokenizer_merges_owt_train_32000.json",
    )

    # qab(ts_tokenizer, owt_tokenizer)
    qc(ts_tokenizer, owt_tokenizer)
    # qd(ts_tokenizer, owt_tokenizer)

######
# ord('a') -> 97
# chr(97) -> 'a'

# unicode1:
# a. chr(0) is return '\x00'
# b. print it return nothing, the `__repr__()` is '\x00' as what we have seen above
# c.
# ```python
# >>> "this is a test" + chr(0) + "string"
# 'this is a test\x00string'
# >>> print("this is a test" + chr(0) + "string")
# this is a teststring
# ```

# unicode 2:
# a. Using utf-8 instead of utf-16 or utf-32, because utf-8 provides shorter int list.
# b. Because the code using `bytes([b]).decode`, which assume that any single bytes can be decodes,
#    but, '你好' 's encode can't directly decode back for just a single bytes.
# ```python
# >>> '你'.encode('utf-8')
# b'\xe4\xbd\xa0'
# >>> wrong('你好'.encode('utf-8'))
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "<stdin>", line 2, in wrong
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
# ```
# c.
# ```python
# >>> bytes([228]).decode('utf-8')
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
# >>> bytes([228, 189]).decode('utf-8')
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data
# >>> bytes([228, 189, 160]).decode('utf-8')
# ```
######

import os
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENDOFTEXT: str = "<|endoftext|>"
ENDOFTEXT_BYTES: bytes = ENDOFTEXT.encode("utf-8")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _process_chunk_for_pretokenization(args: tuple[str, list[str], str]) -> Counter[bytes]:
    """
    Process a single chunk of text for pre-tokenization.
    This function is designed to be called in parallel by multiprocessing.

    Args:
        args: Tuple of (chunk_text, special_tokens, pattern)

    Returns:
        Counter of pre-tokens found in this chunk
    """
    chunk, special_tokens, pattern = args

    pre_tokens_cnt: Counter[bytes] = Counter()
    special_split_re = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None
    pat_re = re.compile(pattern)

    segments = special_split_re.split(chunk) if special_split_re else [chunk]
    for seg in segments:
        if not seg:
            continue
        pre_tokens_cnt.update(m.group(0).encode("utf-8") for m in pat_re.finditer(seg))

    return pre_tokens_cnt


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int = 270,
    special_tokens: list[str] = [ENDOFTEXT],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    pre_tokens_cnt: Counter[bytes] = Counter()
    vocab_lst = _build_init_vocab_lst(special_tokens)

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, ENDOFTEXT_BYTES)

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    chunk_args = [(chunk, special_tokens, PAT) for chunk in chunks]
    with Pool(processes=num_processes) as pool:
        counters = pool.map(_process_chunk_for_pretokenization, chunk_args)
    for counter in counters:
        pre_tokens_cnt.update(counter)

    token_pair_cnt, pre_spilt_tokens = _init_pair_cnt(pre_tokens_cnt)

    while len(vocab_lst) < vocab_size:
        pair_merge = max(token_pair_cnt, key=lambda pair: (token_pair_cnt[pair], pair))
        for pre_token, split_token_lst in pre_spilt_tokens.items():
            freq = pre_tokens_cnt[pre_token]
            merge_token1 = pair_merge[0]
            merge_token2 = pair_merge[1]
            new_token = pair_merge[0] + pair_merge[1]
            if new_token not in pre_token:
                continue
            else:
                i = 0
                while i < len(split_token_lst) - 1:
                    if split_token_lst[i] == merge_token1 and split_token_lst[i + 1] == merge_token2:
                        if i > 0:
                            token_pair_cnt[(split_token_lst[i - 1], split_token_lst[i])] -= freq
                            token_pair_cnt[(split_token_lst[i - 1], new_token)] += freq
                        if i + 2 < len(split_token_lst):
                            token_pair_cnt[(split_token_lst[i + 1], split_token_lst[i + 2])] -= freq
                            token_pair_cnt[(new_token, split_token_lst[i + 2])] += freq
                        split_token_lst[i : i + 2] = [new_token]
                        # not increase i, check new merged
                    else:
                        i += 1
        token_pair_cnt[(pair_merge[0], pair_merge[1])] = 0
        merges.append(pair_merge)
        vocab_lst.append(new_token)

    for idx, token_bytes in enumerate(vocab_lst):
        vocab[idx] = token_bytes
    print(merges)
    return vocab, merges


def _build_init_vocab_lst(special_tokens: list[str]) -> list[bytes]:
    vocab_lst = []
    for token in special_tokens:
        vocab_lst.append(token.encode("utf-8"))
    for i in range(256):
        vocab_lst.append(bytes([i]))
    return vocab_lst


def _init_pair_cnt(pre_tokens_cnt: Counter[bytes]) -> tuple[Counter[tuple[bytes, bytes]], dict[bytes, list[bytes]]]:
    cnt: Counter[tuple[bytes, bytes]] = Counter()
    pre_split_tokens: dict[bytes, list[bytes]] = {}
    for token_byte, freq in pre_tokens_cnt.items():
        split_bytes = [bytes([i]) for i in token_byte]
        for i in range(len(token_byte) - 1):
            pair = (bytes([token_byte[i]]), bytes([token_byte[i + 1]]))
            cnt[pair] += freq
            pre_split_tokens[token_byte] = split_bytes
    return cnt, pre_split_tokens


if __name__ == "__main__":
    train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt")

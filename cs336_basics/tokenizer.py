import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .bpe import PATTERN, get_or_compile_pattern, get_or_compile_special_split_pattern


class Tokenier:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.pattern_re = get_or_compile_pattern(PATTERN)
        self.special_split_re = (
            get_or_compile_special_split_pattern(special_tokens)
            if special_tokens
            else None
        )

        # rank, lower is higher priority
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            (prev, next_): i for i, (prev, next_) in enumerate(merges)
        }
        # inverse map for bytes to token ID, for O(1) lookup
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str = "data/tokenizer_vocab_owt_train_32000.json",
        merges_filepath: str = "data/tokenizer_merges_owt_train_32000.json",
        special_tokens: list[str] | None = None,
    ):
        project_root = Path(__file__).parent.parent
        vocab_path = project_root / vocab_filepath
        merges_path = project_root / merges_filepath

        with (
            open(vocab_path, encoding="utf-8") as vocab_file,
            open(merges_path, encoding="utf-8") as merges_file,
        ):
            vocab = vocab = {
                int(k): bytes.fromhex(v) for k, v in json.load(vocab_file).items()
            }
            merges = [
                (bytes.fromhex(left), bytes.fromhex(right))
                for left, right in json.load(merges_file)
            ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        segments: list[str] = (
            self.special_split_re.split(text) if self.special_split_re else [text]
        )
        vector = []
        for segment in segments:
            if self.special_tokens and segment in self.special_tokens:
                vector.append(self.bytes_to_id.get(segment.encode("utf-8"), -1))
            else:
                pretokens = [
                    match_.group(0).encode("utf-8")
                    for match_ in self.pattern_re.finditer(segment)
                ]
                for pretoken in pretokens:
                    vector.extend(self._bpe_encode(pretoken))
        return vector

    def _bpe_encode(self, pretoken: bytes) -> list[int]:
        bytes_lst = [bytes([b]) for b in pretoken]

        while len(bytes_lst) >= 2:
            new_bytes_lst = self._merge_bytes(bytes_lst)
            if len(new_bytes_lst) == len(
                bytes_lst
            ):  # the length unchanged means no more pairs can be merged
                break
            bytes_lst = new_bytes_lst

        # O(1) because of the inverse hash map
        token_ids: list[int] = []
        for byte_seq in bytes_lst:
            token_id = self.bytes_to_id.get(byte_seq, -1)
            token_ids.append(token_id)
        return token_ids

    def _merge_bytes(self, bytes_lst: list[bytes]) -> list[bytes]:
        """Merge in bytes_lst according to highest merge_ranks."""
        pairs_in_sequence = {}
        for i in range(len(bytes_lst) - 1):
            pair = (bytes_lst[i], bytes_lst[i + 1])
            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]
                # greedy merge the highest priority pair
                if pair not in pairs_in_sequence or rank < pairs_in_sequence[pair]:
                    pairs_in_sequence[pair] = rank

        if not pairs_in_sequence:
            return bytes_lst

        pair_to_merge = min(pairs_in_sequence.keys(), key=lambda p: pairs_in_sequence[p])

        # merge the pair
        result = []
        i = 0
        while i < len(bytes_lst):
            if (
                i < len(bytes_lst) - 1
                and bytes_lst[i] == pair_to_merge[0]
                and bytes_lst[i + 1] == pair_to_merge[1]
            ):
                result.append(bytes_lst[i] + bytes_lst[i + 1])
                i += 2
            else:
                result.append(bytes_lst[i])
                i += 1

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        bytes_seq = b"".join(self.vocab.get(i, b"") for i in ids)
        return bytes_seq.decode("utf-8", errors="replace")


if __name__ == "__main__":
    t = Tokenier.from_files()
    print(
        t.encode(
            "Cassie Lintz settles down for the night with her daughters Kendal, age 4, and Chloe, age 6, on right. "
        )
    )

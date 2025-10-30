import heapq
import itertools
import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
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
        # only cache pretoken with immutable tuple
        self._bpe_encode_cached = lru_cache(maxsize=30000)(
            lambda pretoken: tuple(self._bpe_encode_uncached(pretoken))
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
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
        return list(self._bpe_encode_cached(pretoken))

    def _bpe_encode_uncached(self, pretoken: bytes) -> list[int]:
        """BPE encode a single pretoken (in bytes) without caching.

        Using a double linked list and a min-heap to efficiently perform merges.
        """
        bytes_lst = [bytes([b]) for b in pretoken]
        bytes_node_list = PreTokenDoubleLinkedList()
        for b in bytes_lst:
            bytes_node_list.append(b)

        bytes_heap: list[tuple[int, int, PreTokenNode]] = []  # rank, serial, node
        node = bytes_node_list.head.next
        cnt = itertools.count()
        while node and node.next and node.next != bytes_node_list.tail:
            rank = self.merge_ranks.get((node.val, node.next.val), None)
            if rank is not None:
                bytes_heap.append((rank, next(cnt), node))
            node = node.next
        heapq.heapify(bytes_heap)

        def _push_pair(node: PreTokenNode | None):
            if not node:
                return
            if (
                node.alive
                and node.next
                and node.next.alive
                and node.next != bytes_node_list.tail
            ):
                rank = self.merge_ranks.get((node.val, node.next.val), None)
                if rank is not None:
                    heapq.heappush(bytes_heap, (rank, next(cnt), node))

        while bytes_heap:
            rank, _, node = heapq.heappop(bytes_heap)
            if (
                node.alive
                and node.next
                and node.next.alive
                and self.merge_ranks.get((node.val, node.next.val)) == rank
            ):
                right = node.next
                next_node = right.next

                node.val = node.val + right.val

                right.alive = False
                right.next = None
                right.prev = None

                node.next = next_node
                if next_node:
                    next_node.prev = node

                if node.prev:
                    _push_pair(node.prev)
                if node.next:
                    _push_pair(node)

        lst = []
        node = bytes_node_list.head.next
        while node and node != bytes_node_list.tail:
            if node.alive:
                lst.append(node.val)
            node = node.next
        return [self.bytes_to_id.get(b, -1) for b in lst]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of text strings.

        Yields tokens one at a time without building full token lists in memory.
        """
        for text in iterable:
            yield from self._encode_stream(text)

    def _encode_stream(self, text: str) -> Iterator[int]:
        """
        Internal streaming encoder that yields tokens one at a time.

        This avoids building the full token list in memory, making it
        suitable for processing large texts.
        """
        segments: list[str] = (
            self.special_split_re.split(text) if self.special_split_re else [text]
        )

        for segment in segments:
            if self.special_tokens and segment in self.special_tokens:
                token_id = self.bytes_to_id.get(segment.encode("utf-8"), -1)
                yield token_id
            else:
                # Process pretokens one at a time
                for match_ in self.pattern_re.finditer(segment):
                    pretoken = match_.group(0).encode("utf-8")
                    yield from self._bpe_encode(pretoken)

    def decode(self, ids: list[int]) -> str:
        bytes_seq = b"".join(self.vocab.get(i, b"") for i in ids)
        return bytes_seq.decode("utf-8", errors="replace")


class PreTokenNode:
    def __init__(self, byte: bytes, prev=None, next_=None, alive=True):
        self.val = byte
        self.next: PreTokenNode | None = next_
        self.prev: PreTokenNode | None = prev
        self.alive = alive


class PreTokenDoubleLinkedList:
    def __init__(self):
        self.head: PreTokenNode = PreTokenNode(b"", alive=False)
        self.tail: PreTokenNode = PreTokenNode(b"", alive=False)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def _add_node(
        self,
        new_node: PreTokenNode,
        prev_node: PreTokenNode,
        next_node: PreTokenNode,
    ):
        prev_node.next = new_node
        new_node.prev = prev_node
        new_node.next = next_node
        next_node.prev = new_node
        self.size += 1
        return new_node

    def append(self, byte: bytes) -> PreTokenNode:
        new_node = PreTokenNode(byte)
        # tail.prev is guaranteed to be non-None after __init__
        assert self.tail.prev is not None
        return self._add_node(new_node, self.tail.prev, self.tail)


if __name__ == "__main__":
    t = Tokenier.from_files(
        vocab_filepath="data/tokenizer_vocab_owt_train_32000.json",
        merges_filepath="data/tokenizer_merges_owt_train_32000.json",
    )
    print(
        t.encode(
            "Cassie Lintz settles down for the night with her daughters Kendal, age 4, and Chloe, age 6, on right. "
        )
    )

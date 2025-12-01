"""Byte-pair encoding tokenizer utilities."""

from __future__ import annotations

import heapq
import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

from .bpe_shared import (
    PATTERN,
    get_or_compile_pattern,
    get_or_compile_special_split_pattern,
)


class Tokenier:
    """Tokenizer implementing byte-pair encoding merges with caching support."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Create a tokenizer using the provided vocabulary and merge rules."""
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
            lambda pretoken: tuple(self._bpe_encode_uncached(pretoken)),
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenier:
        """Create a tokenizer instance from serialized vocabulary and merges."""
        project_root = Path(__file__).parent.parent
        vocab_path = project_root / vocab_filepath
        merges_path = project_root / merges_filepath

        with (
            vocab_path.open(encoding="utf-8") as vocab_file,
            merges_path.open(encoding="utf-8") as merges_file,
        ):
            vocab = {int(k): bytes.fromhex(v) for k, v in json.load(vocab_file).items()}
            merges = [
                (bytes.fromhex(left), bytes.fromhex(right))
                for left, right in json.load(merges_file)
            ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a single string into a list of token identifiers."""
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
        bytes_node_list = self._build_pre_token_list(pretoken)
        counter = itertools.count()
        bytes_heap = self._build_initial_heap(bytes_node_list, counter)
        self._merge_pairs(bytes_node_list, bytes_heap, counter)
        return [
            self.bytes_to_id.get(b, -1)
            for b in self._collect_byte_sequences(bytes_node_list)
        ]

    def _build_pre_token_list(self, pretoken: bytes) -> PreTokenDoubleLinkedList:
        """Convert a byte sequence into a linked list representation."""
        bytes_node_list = PreTokenDoubleLinkedList()
        for value in pretoken:
            bytes_node_list.append(bytes([value]))
        return bytes_node_list

    def _build_initial_heap(
        self,
        bytes_node_list: PreTokenDoubleLinkedList,
        counter: Iterator[int],
    ) -> list[tuple[int, int, PreTokenNode]]:
        """Populate the initial heap with mergeable byte pairs."""
        bytes_heap: list[tuple[int, int, PreTokenNode]] = []
        node = bytes_node_list.head.next
        while node and node.next and node.next != bytes_node_list.tail:
            rank = self.merge_ranks.get((node.val, node.next.val))
            if rank is not None:
                bytes_heap.append((rank, next(counter), node))
            node = node.next
        heapq.heapify(bytes_heap)
        return bytes_heap

    def _merge_pairs(
        self,
        bytes_node_list: PreTokenDoubleLinkedList,
        bytes_heap: list[tuple[int, int, PreTokenNode]],
        counter: Iterator[int],
    ) -> None:
        """Apply merge operations until no eligible pairs remain."""
        tail = bytes_node_list.tail
        while bytes_heap:
            rank, _, node = heapq.heappop(bytes_heap)
            if not self._should_merge(node, rank):
                continue

            right = node.next
            if right is None:
                continue
            next_node = right.next

            node.val = node.val + right.val

            right.alive = False
            right.next = None
            right.prev = None

            node.next = next_node
            if next_node:
                next_node.prev = node

            if node.prev:
                self._push_pair(bytes_heap, counter, node.prev, tail)
            self._push_pair(bytes_heap, counter, node, tail)

    def _should_merge(self, node: PreTokenNode, rank: int) -> bool:
        """Return whether the given node should be merged."""
        return (
            node.alive
            and node.next is not None
            and node.next.alive
            and self.merge_ranks.get((node.val, node.next.val)) == rank
        )

    def _push_pair(
        self,
        bytes_heap: list[tuple[int, int, PreTokenNode]],
        counter: Iterator[int],
        node: PreTokenNode | None,
        tail: PreTokenNode,
    ) -> None:
        """Push an eligible pair onto the heap."""
        if (
            node
            and node.alive
            and node.next
            and node.next.alive
            and node.next is not tail
        ):
            rank = self.merge_ranks.get((node.val, node.next.val))
            if rank is not None:
                heapq.heappush(bytes_heap, (rank, next(counter), node))

    def _collect_byte_sequences(
        self,
        bytes_node_list: PreTokenDoubleLinkedList,
    ) -> list[bytes]:
        """Collect contiguous byte sequences from the linked list."""
        sequences: list[bytes] = []
        node = bytes_node_list.head.next
        while node and node != bytes_node_list.tail:
            if node.alive:
                sequences.append(node.val)
            node = node.next
        return sequences

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Memory-efficient encoding of an iterable of text strings.

        Yields tokens one at a time without building full token lists in memory.
        """
        for text in iterable:
            yield from self._encode_stream(text)

    def _encode_stream(self, text: str) -> Iterator[int]:
        """Yield tokens one at a time during streaming encoding.

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
        """Decode a sequence of token identifiers into a UTF-8 string."""
        bytes_seq = b"".join(self.vocab.get(i, b"") for i in ids)
        return bytes_seq.decode("utf-8", errors="replace")


class PreTokenNode:
    """A node in the double linked list representing a pretoken sequence."""

    def __init__(
        self,
        byte: bytes,
        prev: PreTokenNode | None = None,
        next_: PreTokenNode | None = None,
        *,
        alive: bool = True,
    ) -> None:
        """Initialize a node representing a byte in the merge structure."""
        self.val = byte
        self.next: PreTokenNode | None = next_
        self.prev: PreTokenNode | None = prev
        self.alive = alive


class PreTokenDoubleLinkedList:
    """Double linked list used to support efficient byte-pair merges."""

    def __init__(self) -> None:
        """Create an empty linked list with sentinel head and tail nodes."""
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
    ) -> PreTokenNode:
        """Insert ``new_node`` between ``prev_node`` and ``next_node``."""
        prev_node.next = new_node
        new_node.prev = prev_node
        new_node.next = next_node
        next_node.prev = new_node
        self.size += 1
        return new_node

    def append(self, byte: bytes) -> PreTokenNode:
        """Append a new byte to the list and return the created node."""
        new_node = PreTokenNode(byte)
        prev_tail = self.tail.prev
        if prev_tail is None:
            msg = "PreTokenDoubleLinkedList tail missing predecessor."
            raise RuntimeError(msg)
        return self._add_node(new_node, prev_tail, self.tail)


if __name__ == "__main__":
    t = Tokenier.from_files(
        vocab_filepath="data/tokenizer_vocab_owt_train_32000.json",
        merges_filepath="data/tokenizer_merges_owt_train_32000.json",
    )
    print(
        t.encode(
            (
                "Cassie Lintz settles down for the night with her daughters Kendal, "
                "age 4, and Chloe, age 6, on right. "
            ),
        ),
    )

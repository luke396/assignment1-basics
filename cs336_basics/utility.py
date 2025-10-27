def _format_bytes(b: bytes) -> str:
    """Return a readable representation of bytes: repr plus hex string."""
    try:
        # Keep repr for escapes, also provide hex for clarity
        return f"{b!r} (hex={b.hex()})"
    except Exception:
        return repr(b)


def print_bpe_result(
    bpe_result: tuple[dict[int, bytes], list[tuple[bytes, bytes]]] | None = None,
):
    assert bpe_result is not None, "No BPE result to print"
    vocab, merge = bpe_result
    _print_vocab_sample(vocab)
    _print_merges_sample(merge)


def _print_vocab_sample(vocab: dict[int, bytes], limit: int = 20) -> None:
    keys = sorted(vocab.keys())
    limit = max(0, int(limit))
    take = keys[-limit:] if limit else []
    print(f"Vocab sample (last {len(take)} of {len(keys)}):")
    for idx in take:
        print(f"  {idx}: {_format_bytes(vocab[idx])}")
    if vocab:
        # Longest by byte length
        top_k = min(5, len(vocab))
        longest_items = sorted(vocab.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top_k]
        print(f"Top {top_k} longest tokens (by bytes):")
        for rank, (tok_id, tok_bytes) in enumerate(longest_items, start=1):
            print(
                f"  {rank}) id={tok_id}, len={len(tok_bytes)} bytes, value={_format_bytes(tok_bytes)}"
            )


def _print_merges_sample(merges: list[tuple[bytes, bytes]], limit: int = 20) -> None:
    take = merges[:limit]
    print(f"Merges sample (first {len(take)} of {len(merges)}):")
    for i, (a, b) in enumerate(take):
        merged = a + b
        print(
            f"  {i}: {_format_bytes(a)} + {_format_bytes(b)} -> {_format_bytes(merged)}"
        )

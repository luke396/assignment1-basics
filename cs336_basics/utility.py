import gzip
from pathlib import Path

import msgpack


def _format_bytes(b: bytes) -> str:
    """Return a readable representation of bytes: repr plus hex string."""
    try:
        # Keep repr for escapes, also provide hex for clarity
        return f"{b!r} (hex={b.hex()})"
    except Exception:
        return repr(b)


def print_bpe_result(
    output_path: Path | None = None, bpe_result: tuple[dict[int, bytes], list[tuple[bytes, bytes]]] | None = None
):
    if output_path:
        vocab, merge = load_bpe_msgpack(output_path)
    elif bpe_result:
        vocab, merge = bpe_result
    else:
        raise (ValueError("Nothing to print."))

    print_vocab_sample(vocab)
    print_merges_sample(merge)


def print_vocab_sample(vocab: dict[int, bytes], limit: int = 20) -> None:
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
            print(f"  {rank}) id={tok_id}, len={len(tok_bytes)} bytes, value={_format_bytes(tok_bytes)}")


def print_merges_sample(merges: list[tuple[bytes, bytes]], limit: int = 20) -> None:
    take = merges[:limit]
    print(f"Merges sample (first {len(take)} of {len(merges)}):")
    for i, (a, b) in enumerate(take):
        merged = a + b
        print(f"  {i}: {_format_bytes(a)} + {_format_bytes(b)} -> {_format_bytes(merged)}")


def save_bpe_msgpack(
    vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], out_path: str | Path, compress: bool = True
):
    payload = {
        "format": "bpe-msgpack-v1",
        "vocab": vocab,  # dict[int, bytes]
        "merges": merges,  # list[tuple[bytes, bytes]]
    }
    out_path = Path(out_path)

    data = msgpack.packb(payload, use_bin_type=True)
    if data is None:
        raise RuntimeError("msgpack.packb returned None")
    if compress:
        with gzip.open(out_path, "wb") as f:
            f.write(data)
    else:
        with open(out_path, "wb") as f:
            f.write(data)


def load_bpe_msgpack(path: str | Path):
    path = Path(path)
    opener = gzip.open if path.suffix.endswith("gz") else open
    with opener(path, "rb") as f:
        # strict_map_key=False allows int keys in maps (our vocab uses int indices)
        payload = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    fmt = payload.get("format", "bpe-msgpack")
    if fmt not in {"bpe-msgpack", "bpe-msgpack-v1"}:
        raise ValueError(f"Unsupported BPE serialization format: {fmt}")
    vocab = {int(k): bytes(v) for k, v in payload["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in payload["merges"]]
    return vocab, merges

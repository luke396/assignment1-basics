"""Test paged attention: compare against standard attention for correctness."""

import torch

from cs336_basics.paged_attention import (
    KVCacheManager,
    paged_attention_decode,
)


def standard_attention(
    q: torch.Tensor,
    k: list[torch.Tensor],
    v: list[torch.Tensor],
) -> torch.Tensor:
    """Compute standard scaled dot-product attention for reference.

    Args:
        q: (num_seqs, num_heads, head_dim) — one query token per seq.
        k: list of tensors, each (seq_len_i, num_heads, head_dim).
        v: list of tensors, each (seq_len_i, num_heads, head_dim).

    Returns:
        Tensor of shape (num_seqs, num_heads, head_dim).

    """
    outputs = []
    scale = q.shape[-1] ** -0.5
    for i in range(q.shape[0]):
        qi = q[i]  # (num_heads, head_dim)
        ki = k[i]  # (seq_len, num_heads, head_dim)
        vi = v[i]  # (seq_len, num_heads, head_dim)
        # (num_heads, head_dim) @ (num_heads, head_dim, seq_len) -> (num_heads, seq_len)
        scores = torch.einsum("h d, t h d -> h t", qi, ki) * scale
        weights = torch.softmax(scores, dim=-1)  # (num_heads, seq_len)
        out = torch.einsum("h t, t h d -> h d", weights, vi)  # (num_heads, head_dim)
        outputs.append(out)
    return torch.stack(outputs)  # (num_seqs, num_heads, head_dim)


def test_basic_correctness() -> None:
    """Test 1: paged attention output matches standard attention."""
    print("Test 1: Basic correctness (batch=2, variable seq_lens) ...", end=" ")

    torch.manual_seed(42)
    num_heads, head_dim, block_size = 4, 8, 4
    seq_lens_list = [10, 6]  # two sequences with different lengths
    num_seqs = len(seq_lens_list)
    num_blocks = 32

    manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim)

    # Generate random K/V for each sequence and write into paged cache
    all_k, all_v = [], []
    seq_ids = list(range(num_seqs))
    for sid, slen in zip(seq_ids, seq_lens_list, strict=True):
        manager.register_sequence(sid)
        k_tokens = torch.randn(slen, num_heads, head_dim)
        v_tokens = torch.randn(slen, num_heads, head_dim)
        all_k.append(k_tokens)
        all_v.append(v_tokens)
        for t in range(slen):
            manager.append_token(sid, k_tokens[t], v_tokens[t])

    # Random query (decode: 1 token per seq)
    q = torch.randn(num_seqs, num_heads, head_dim)

    # Ground truth: standard attention
    expected = standard_attention(q, all_k, all_v)

    # Paged attention
    block_tables, seq_lens = manager.build_block_tables(seq_ids)
    actual = paged_attention_decode(
        q,
        manager.kv_cache.key_cache,
        manager.kv_cache.value_cache,
        block_tables,
        seq_lens,
    )

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Mismatch!\nmax diff: {(expected - actual).abs().max().item()}"
    )
    print("PASSED")


def test_single_token() -> None:
    """Test 2: edge case — sequence with only 1 token."""
    print("Test 2: Single token sequence ...", end=" ")

    torch.manual_seed(123)
    num_heads, head_dim, block_size = 2, 4, 4
    num_blocks = 8

    manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim)
    manager.register_sequence(0)

    k = torch.randn(num_heads, head_dim)
    v = torch.randn(num_heads, head_dim)
    manager.append_token(0, k, v)

    q = torch.randn(1, num_heads, head_dim)

    expected = standard_attention(q, [k.unsqueeze(0)], [v.unsqueeze(0)])
    block_tables, seq_lens = manager.build_block_tables([0])
    actual = paged_attention_decode(
        q,
        manager.kv_cache.key_cache,
        manager.kv_cache.value_cache,
        block_tables,
        seq_lens,
    )

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Mismatch!\nmax diff: {(expected - actual).abs().max().item()}"
    )
    print("PASSED")


def test_exact_block_boundary() -> None:
    """Test 3: seq_len is exact multiple of block_size (no partial last block)."""
    print("Test 3: Exact block boundary (seq_len=8, block_size=4) ...", end=" ")

    torch.manual_seed(77)
    num_heads, head_dim, block_size = 4, 8, 4
    seq_len = 8  # exactly 2 blocks
    num_blocks = 16

    manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim)
    manager.register_sequence(0)

    k_tokens = torch.randn(seq_len, num_heads, head_dim)
    v_tokens = torch.randn(seq_len, num_heads, head_dim)
    for t in range(seq_len):
        manager.append_token(0, k_tokens[t], v_tokens[t])

    q = torch.randn(1, num_heads, head_dim)

    expected = standard_attention(q, [k_tokens], [v_tokens])
    block_tables, seq_lens = manager.build_block_tables([0])
    actual = paged_attention_decode(
        q,
        manager.kv_cache.key_cache,
        manager.kv_cache.value_cache,
        block_tables,
        seq_lens,
    )

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Mismatch!\nmax diff: {(expected - actual).abs().max().item()}"
    )
    print("PASSED")


def test_cow_fork() -> None:
    """Test 4: CoW fork — two sequences share blocks, then diverge."""
    print("Test 4: Copy-on-Write fork ...", end=" ")

    torch.manual_seed(99)
    num_heads, head_dim, block_size = 4, 8, 4
    prompt_len = 6
    num_blocks = 32

    manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim)
    manager.register_sequence(0)

    # Shared prompt
    k_prompt = torch.randn(prompt_len, num_heads, head_dim)
    v_prompt = torch.randn(prompt_len, num_heads, head_dim)
    for t in range(prompt_len):
        manager.append_token(0, k_prompt[t], v_prompt[t])

    # Fork: seq 1 shares blocks with seq 0
    manager.fork_sequence(0, 1)

    # Verify shared block IDs
    blocks_0 = manager.get_block_ids(0)
    blocks_1 = manager.get_block_ids(1)
    assert blocks_0 == blocks_1, "Forked sequences should share block IDs"

    # Each sequence appends a different token (triggers CoW on last block)
    k_new_0 = torch.randn(num_heads, head_dim)
    v_new_0 = torch.randn(num_heads, head_dim)
    k_new_1 = torch.randn(num_heads, head_dim)
    v_new_1 = torch.randn(num_heads, head_dim)

    manager.append_token(0, k_new_0, v_new_0)
    manager.append_token(1, k_new_1, v_new_1)

    # After CoW, last blocks should differ
    blocks_0_after = manager.get_block_ids(0)
    blocks_1_after = manager.get_block_ids(1)
    assert blocks_0_after[-1] != blocks_1_after[-1], (
        "Last block should differ after CoW"
    )
    # But earlier blocks still shared
    assert blocks_0_after[:-1] == blocks_1_after[:-1], (
        "Earlier blocks should still be shared"
    )

    # Verify attention output for both sequences
    q = torch.randn(2, num_heads, head_dim)

    all_k_0 = torch.cat([k_prompt, k_new_0.unsqueeze(0)])
    all_v_0 = torch.cat([v_prompt, v_new_0.unsqueeze(0)])
    all_k_1 = torch.cat([k_prompt, k_new_1.unsqueeze(0)])
    all_v_1 = torch.cat([v_prompt, v_new_1.unsqueeze(0)])

    expected = standard_attention(q, [all_k_0, all_k_1], [all_v_0, all_v_1])
    block_tables, seq_lens = manager.build_block_tables([0, 1])
    actual = paged_attention_decode(
        q,
        manager.kv_cache.key_cache,
        manager.kv_cache.value_cache,
        block_tables,
        seq_lens,
    )

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Mismatch!\nmax diff: {(expected - actual).abs().max().item()}"
    )
    print("PASSED")


def main() -> None:
    """Run all paged attention tests."""
    test_basic_correctness()
    test_single_token()
    test_exact_block_boundary()
    test_cow_fork()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()

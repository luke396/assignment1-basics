"""Demo: continuous batching vs static batching.

Constructs a small TransformerLM, sends multiple requests arriving at different
steps, and prints per-step batch composition.  At the end it compares total
steps against a naive static-batching baseline and verifies KV cache blocks
are fully reclaimed.
"""

from collections import deque

import torch

from cs336_basics.blocks import RotaryPositionalEmbedding, TransformerLM
from cs336_basics.continuous_batching import (
    ContinuousBatchingEngine,
    Scheduler,
    Sequence,
    SequenceStatus,
)
from cs336_basics.paged_attention import KVCacheManager

# ── tiny model config ────────────────────────────────────────────────
VOCAB_SIZE = 128
D_MODEL = 64
NUM_HEADS = 4
D_FF = 128
CTX_LEN = 256
N_LAYERS = 2
BLOCK_SIZE = 4
NUM_BLOCKS = 64
MAX_CONCURRENT = 3
EOS_TOKEN_ID = VOCAB_SIZE - 1  # 127


def build_model() -> TransformerLM:
    rope = RotaryPositionalEmbedding(
        theta=10000.0,
        d_k=D_MODEL // NUM_HEADS,
        max_seq_len=CTX_LEN,
    )
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        context_length=CTX_LEN,
        n_layers=N_LAYERS,
        rope=rope,
    )
    model.requires_grad_(False)
    return model


def make_requests() -> list[tuple[int, Sequence]]:
    """Create 5 requests arriving at different steps with varied prompt lengths."""
    torch.manual_seed(0)
    requests = [
        # (arrival_step, seq_id, prompt_length, max_new_tokens)
        (0, 0, 8, 6),
        (0, 1, 5, 4),
        (1, 2, 10, 5),
        (3, 3, 4, 3),
        (4, 4, 6, 7),
    ]
    seqs = []
    for arrival, sid, plen, mnt in requests:
        prompt = torch.randint(0, VOCAB_SIZE - 1, (plen,)).tolist()
        seq = Sequence(
            seq_id=sid,
            status=SequenceStatus.WAITING,
            prompt_tokens=prompt,
            output_tokens=[],
            max_new_tokens=mnt,
        )
        seqs.append((arrival, seq))
    return seqs


def build_engine(model: TransformerLM) -> tuple[ContinuousBatchingEngine, KVCacheManager]:
    head_dim = D_MODEL // NUM_HEADS
    kv_manager = KVCacheManager(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
        head_dim=head_dim,
        n_layers=N_LAYERS,
    )
    scheduler = Scheduler(
        max_concurrent=MAX_CONCURRENT,
        block_size=BLOCK_SIZE,
        free_block_num_fn=kv_manager.allocator.num_free_blocks,
    )
    engine = ContinuousBatchingEngine(
        model=model,
        kv_manager=kv_manager,
        scheduler=scheduler,
        eos_token_id=EOS_TOKEN_ID,
    )
    return engine, kv_manager


def run_continuous_batching(
    engine: ContinuousBatchingEngine,
    kv_manager: KVCacheManager,
    requests: deque[tuple[int, Sequence]],
) -> tuple[list[Sequence], int]:
    """Run engine loop with per-step logging. Returns (finished, total_steps)."""
    step = 0
    initial_free = kv_manager.allocator.num_free_blocks()
    print(f"  Free blocks at start: {initial_free}")
    print()

    while requests or engine.scheduler.has_unfinished():
        # inject arrivals for this step
        while requests and requests[0][0] == step:
            _, seq = requests.popleft()
            engine.scheduler.add_request(seq)
            print(f"  Step {step}: ++ inject Seq {seq.seq_id} "
                  f"(prompt_len={len(seq.prompt_tokens)}, "
                  f"max_new={seq.max_new_tokens})")

        output = engine.scheduler.schedule()

        # free finished
        for seq_id in output.freed_seq_ids:
            kv_manager.free_sequence(seq_id)

        prefill_ids = [s.seq_id for s in output.prefill_seq]
        decode_ids = [s.seq_id for s in output.decode_seq]
        freed_ids = output.freed_seq_ids
        free_now = kv_manager.allocator.num_free_blocks()

        parts = []
        if prefill_ids:
            parts.append(f"prefill={prefill_ids}")
        if decode_ids:
            parts.append(f"decode={decode_ids}")
        if freed_ids:
            parts.append(f"freed={freed_ids}")
        batch_str = ", ".join(parts) if parts else "idle"
        print(f"  Step {step}: {batch_str}  (free_blocks={free_now})")

        with torch.inference_mode():
            for seq in output.prefill_seq:
                engine._prefill(seq)
            if output.decode_seq:
                engine._decode_batch(output.decode_seq)

        step += 1

    final_free = kv_manager.allocator.num_free_blocks()
    print(f"\n  Free blocks at end: {final_free}")
    return engine.scheduler.finished, step


def static_batching_steps(requests: list[tuple[int, Sequence]]) -> int:
    """Estimate total steps for static batching.

    Static batching waits until all requests in a batch arrive, then processes
    them together.  Every sequence must wait for the longest one to finish
    before the batch slot is freed.  We simulate this by grouping requests
    by arrival step and processing each group as a batch.
    """
    from itertools import groupby

    total = 0
    for _, group in groupby(requests, key=lambda r: r[0]):
        batch = list(group)
        max_new = max(seq.max_new_tokens for _, seq in batch)
        total += 1 + max_new  # 1 prefill step + max_new decode steps
    return total


def greedy_generate_naive(
    model: TransformerLM, prompt_tokens: list[int], max_new_tokens: int, eos_id: int,
) -> list[int]:
    """Generate tokens using naive KV cache (single sequence, greedy argmax)."""
    model.clear_kv_cache()
    input_t = torch.tensor([prompt_tokens])  # (1, prompt_len)
    with torch.inference_mode():
        logits = model(input_t, use_cache=True)  # (1, prompt_len, vocab)
        token = int(torch.argmax(logits[:, -1, :]).item())
        output = [token]
        for _ in range(max_new_tokens - 1):
            if token == eos_id:
                break
            pos = torch.tensor([len(prompt_tokens) + len(output) - 1])
            logits = model(
                torch.tensor([[token]]), token_positions=pos, use_cache=True,
            )
            token = int(torch.argmax(logits[:, -1, :]).item())
            output.append(token)
    return output


def test_correctness(model: TransformerLM):
    """Verify single-request continuous batching matches naive KV cache generation."""
    print("--- Correctness Test: continuous batching vs naive KV cache ---")
    torch.manual_seed(0)
    prompt = torch.randint(0, VOCAB_SIZE - 1, (12,)).tolist()
    max_new = 10

    # naive KV cache path
    naive_output = greedy_generate_naive(model, prompt, max_new, EOS_TOKEN_ID)

    # continuous batching path (single request at step 0)
    engine, kv_manager = build_engine(model)
    seq = Sequence(
        seq_id=0,
        status=SequenceStatus.WAITING,
        prompt_tokens=prompt,
        output_tokens=[],
        max_new_tokens=max_new,
    )
    with torch.inference_mode():
        finished = engine.run(deque([(0, seq)]))
    cb_output = finished[0].output_tokens

    match = naive_output == cb_output
    print(f"  Prompt:     {prompt}")
    print(f"  Naive:      {naive_output}")
    print(f"  Continuous: {cb_output}")
    print(f"  Match: {match}")
    assert match, "Output mismatch between naive KV cache and continuous batching!"
    print()


def main():
    print("=" * 60)
    print("  Continuous Batching Demo")
    print("=" * 60)

    torch.manual_seed(42)
    model = build_model()

    # ── correctness test ────────────────────────────────────────
    test_correctness(model)

    # ── continuous batching ──────────────────────────────────────
    print("\n--- Continuous Batching ---")
    engine, kv_manager = build_engine(model)
    raw_requests = make_requests()

    print("\n  Requests:")
    for arrival, seq in raw_requests:
        print(f"    Seq {seq.seq_id}: arrive@step={arrival}, "
              f"prompt_len={len(seq.prompt_tokens)}, "
              f"max_new={seq.max_new_tokens}")
    print()

    requests_deque = deque(raw_requests)
    finished, total_steps = run_continuous_batching(engine, kv_manager, requests_deque)

    # verify blocks reclaimed
    free_after = kv_manager.allocator.num_free_blocks()
    assert free_after == NUM_BLOCKS, (
        f"Block leak! Expected {NUM_BLOCKS} free, got {free_after}"
    )
    print(f"  All {NUM_BLOCKS} blocks reclaimed.")

    # ── results ─────────────────────────────────────────────────
    print(f"\n  Continuous batching total steps: {total_steps}")
    print(f"\n  Finished sequences:")
    for seq in finished:
        print(f"    Seq {seq.seq_id}: {len(seq.output_tokens)} tokens generated "
              f"(output={seq.output_tokens})")

    # ── static batching comparison ──────────────────────────────
    static_steps = static_batching_steps(make_requests())
    print(f"\n--- Comparison ---")
    print(f"  Static  batching steps (estimate): {static_steps}")
    print(f"  Continuous batching steps:          {total_steps}")
    print()


if __name__ == "__main__":
    main()

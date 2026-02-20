"""Benchmark KV cache: speed comparison and output consistency."""

import time

import torch

from cs336_basics.blocks import RotaryPositionalEmbedding, TransformerLM
from cs336_basics.training_utility import generate


class DummyTokenizer:
    """Minimal tokenizer for benchmarking (identity mapping over ints)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.eot_token_id = vocab_size - 1  # last token as EOT

    def encode(self, text: str) -> list[int]:
        return [int(c) % (self.vocab_size - 1) for c in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i % 128) for i in ids)


MODEL_CONFIG = {
    "vocab_size": 512,
    "context_length": 1024,
    "d_model": 768,
    "num_heads": 12,
    "d_ff": 2048,
    "n_layers": 12,
}


def build_model(config: dict, device: torch.device) -> TransformerLM:
    rope = RotaryPositionalEmbedding(
        theta=10000.0,
        d_k=config["d_model"] // config["num_heads"],
        max_seq_len=config["context_length"],
        device=device,
    )
    return TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        context_length=config["context_length"],
        n_layers=config["n_layers"],
        rope=rope,
        device=device,
    )


def run_generate(
    model: TransformerLM,
    tokenizer: DummyTokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    use_cache: bool,
    seed: int = 42,
) -> tuple[str, float, int]:
    """Returns (text, elapsed_seconds, num_generated_tokens)."""
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    text = generate(
        model,
        tokenizer,  # pyright: ignore[reportArgumentType]
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        rng=rng,
        context_length=context_length,
        use_cache=use_cache,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    num_generated = len(tokenizer.encode(text))
    return text, elapsed, num_generated


def main():
    # Force CPU: GPU kernel launch overhead masks the computation savings
    device = torch.device("cpu")
    config = MODEL_CONFIG
    prompt = "Hello world, this is a benchmark test"
    max_new_tokens = 200

    print(f"Device: {device}")
    print(f"Model: d={config['d_model']}, heads={config['num_heads']}, "
          f"layers={config['n_layers']}, ctx={config['context_length']}")

    model = build_model(config, device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    tokenizer = DummyTokenizer(vocab_size=config["vocab_size"])
    ctx_len = config["context_length"]

    # Warmup
    print("\nWarming up...")
    run_generate(model, tokenizer, prompt, 4, ctx_len, use_cache=False)
    run_generate(model, tokenizer, prompt, 4, ctx_len, use_cache=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- No cache ---
    print(f"\n{'='*50}")
    print(f"  Without KV Cache (max_new_tokens={max_new_tokens})")
    print(f"{'='*50}")
    text_no, t_no, n_no = run_generate(
        model, tokenizer, prompt, max_new_tokens, ctx_len,
        use_cache=False, seed=42,
    )
    tps_no = n_no / t_no
    print(f"Time: {t_no:.2f}s | {tps_no:.1f} tokens/sec")
    print(f"Generated {n_no} tokens")

    # --- With cache ---
    print(f"\n{'='*50}")
    print(f"  With KV Cache (max_new_tokens={max_new_tokens})")
    print(f"{'='*50}")
    text_yes, t_yes, n_yes = run_generate(
        model, tokenizer, prompt, max_new_tokens, ctx_len,
        use_cache=True, seed=42,
    )
    tps_yes = n_yes / t_yes
    print(f"Time: {t_yes:.2f}s | {tps_yes:.1f} tokens/sec")
    print(f"Generated {n_yes} tokens")

    # --- Summary ---
    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    match = text_no == text_yes
    speedup = t_no / t_yes if t_yes > 0 else float("inf")
    print(f"Outputs match: {match}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"No cache: {t_no:.2f}s ({tps_no:.1f} tok/s)")
    print(f"   Cache: {t_yes:.2f}s ({tps_yes:.1f} tok/s)")

    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory: {max_mem:.2f} GB")


if __name__ == "__main__":
    main()

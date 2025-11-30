"""Experiments for blocks.py."""

from .blocks import RotaryPositionalEmbedding, TransformerLM

GPT2XL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 48,
    "d_model": 1600,
    "n_heads": 25,
    "d_ff": 6400,
}
ROPE = RotaryPositionalEmbedding(
    theta=100000.0,
    d_k=GPT2XL_CONFIG["d_model"] // GPT2XL_CONFIG["n_heads"],
    max_seq_len=GPT2XL_CONFIG["context_length"],
)


def qa(config=GPT2XL_CONFIG):
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        embedding_dim=config["d_model"],  # embedding_dim same as d_model
        d_model=config["d_model"],
        num_heads=config["n_heads"],
        d_ff=config["d_ff"],
        context_length=config["context_length"],
        n_layers=config["num_layers"],
        rope=ROPE,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    calculated_params = (
        2 * config["vocab_size"] * config["d_model"]
        + config["num_layers"]
        * (
            4 * config["d_model"] ** 2
            + 3 * config["d_model"] * config["d_ff"]
            + 2 * config["d_model"]
        )
        + config["d_model"]
    )
    print(
        f"Total parameters calculated: {calculated_params}, "
        f"about {calculated_params / 1e9:.2f}B"
    )
    assert total_params == calculated_params

    memory_mb = total_params * 4 / (1024**2)
    memory_gb = memory_mb / 1024
    print(
        f"For single-precision point, FP32 - 4 bytes, "
        f"the memory requirement is: {memory_mb:.2f} MB, "
        f"about {memory_gb:.2f} GB"
    )


def qbc(config=GPT2XL_CONFIG):
    _analyze_flops(config)


def _analyze_flops(config):
    flops_multiattn = (
        8 * config["d_model"] ** 2 + 4 * config["d_model"] * config["context_length"]
    )
    flops_ffn = 4 * config["d_model"] * config["d_ff"]
    flops_n_layers = config["num_layers"] * (flops_multiattn + flops_ffn)
    flops_final_linear = 2 * config["d_model"] * config["vocab_size"]
    total_seq_flops = (flops_n_layers + flops_final_linear) * config["context_length"]
    print(
        "For single batch with seq = context_length,",
        f"for one token, the multiattn FLOPs: {flops_multiattn / 1e9:.2f} GFLOPs, ",
        f"ffn FLOPs: {flops_ffn / 1e9:.2f} GFLOPs, ",
        f"n_layers FLOPs: {flops_n_layers / 1e9:.2f} GFLOPs, ",
        f"final linear FLOPs: {flops_final_linear / 1e9:.2f} GFLOPs, ",
        f"Total seq FLOPs: {total_seq_flops / 1e12:.2f} TFLOPs",
    )

    per_n_layers = flops_n_layers / (flops_n_layers + flops_final_linear)
    per_final_linear = flops_final_linear / (flops_n_layers + flops_final_linear)

    print(
        f"Percentage of n_layers FLOPs: {per_n_layers:.4f}, "
        f"final linear FLOPs: {per_final_linear:.4f}"
    )


def qd(base_config=GPT2XL_CONFIG):
    small = base_config.copy()
    small["n_layers"] = 12
    small["d_model"] = 768
    small["n_heads"] = 12
    print("Small model FLOPs analysis:")
    _analyze_flops(small)
    medium = base_config.copy()
    medium["n_layers"] = 24
    medium["d_model"] = 1024
    medium["n_heads"] = 16
    print("Medium model FLOPs analysis:")
    _analyze_flops(medium)
    large = base_config.copy()
    large["n_layers"] = 36
    large["d_model"] = 1280
    large["n_heads"] = 20
    print("Large model FLOPs analysis:")
    _analyze_flops(large)
    xl_large = base_config.copy()
    xl_large["n_layers"] = 48
    xl_large["d_model"] = 1600
    xl_large["n_heads"] = 25
    print("XLarge model FLOPs analysis:")
    _analyze_flops(xl_large)


def qe(base_config=GPT2XL_CONFIG):
    config = base_config.copy()
    config["context_length"] = 16384
    print("Context length 16K FLOPs analysis:")
    _analyze_flops(config)


if __name__ == "__main__":
    qa()
    qbc()
    qd()
    qe()

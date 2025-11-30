import math
from collections.abc import Callable
from typing import Optional

import torch

GPT2XL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "n_layers": 48,
    "d_model": 1600,
    "n_heads": 25,
    "d_ff": 6400,
}

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def learning_rate_tuning(lr_list):
    for lr in lr_list:
        print(f"Learning rate: {lr}")
        _common_train(lr)
        print("=" * 20)
    print(
        "All losses are decrease. But, the speed of decrease varies, "
        "depending on the learning rate."
    )


def _common_train(lr):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        if t == 0:
            init_loss = loss.cpu().item()
        if t == 99:
            last_loss = loss.cpu().item()
        if t % 10 == 0:
            print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
    print(
        f"Initial loss: {init_loss}, Final loss: {last_loss}",
        f"Decrease:{(init_loss - last_loss):.4f}", # pyright: ignore[reportOperatorIssue]
    )

def adamwAccounting(base_config = GPT2XL_CONFIG):
    xl_large = base_config.copy()
    def _compute_f32memory_coefficients(xl_cfg):
        base = 4 * (
            12 * xl_cfg["n_layers"] * xl_cfg["d_model"] ** 2
            + 2 * xl_cfg["vocab_size"] * xl_cfg["d_model"]
        )
        per_batch = (
            xl_cfg["n_layers"]
            * (
                16 * xl_cfg["d_model"] * xl_cfg["context_length"]
                + xl_cfg["n_heads"] * xl_cfg["context_length"] ** 2
            )
            + xl_cfg["context_length"] * xl_cfg["d_model"]
            + 2 * xl_cfg["context_length"] * xl_cfg["vocab_size"]
        )
        a_bytes = 4 * per_batch
        b_bytes = 4 * base
        return a_bytes, b_bytes

    def _estimate_f32memory(xl_cfg, batch_size=1, coeffs=None):
        a_bytes, b_bytes = coeffs or _compute_f32memory_coefficients(xl_cfg)
        f32memory = a_bytes * batch_size + b_bytes  # a * batch_size + b

        print(
            f"For batch size {batch_size}, estimated float32 memory usage: {f32memory / 1e9:.2f} GB"
        )

    a_bytes, b_bytes = _compute_f32memory_coefficients(xl_large)
    print(
        f"f32memory (GB) = {a_bytes / 1e9:.2f} * batch_size + {b_bytes / 1e9:.2f}, means batch_size increases 1, the memory increases by {a_bytes / 1e9:.2f} GB"
    )

    _estimate_f32memory(xl_large, batch_size=1, coeffs=(a_bytes, b_bytes))
    _estimate_f32memory(xl_large, batch_size=4, coeffs=(a_bytes, b_bytes))
    _estimate_f32memory(xl_large, batch_size=6, coeffs=(a_bytes, b_bytes))
    _estimate_f32memory(xl_large, batch_size=8, coeffs=(a_bytes, b_bytes))

    def _flops(config=GPT2XL_CONFIG,batch_size=1):
        flops_multiattn = (
        8 * config["d_model"] ** 2 + 4 * config["d_model"] * config["context_length"]
    )
        flops_ffn = 4 * config["d_model"] * config["d_ff"]
        flops_n_layers = config["n_layers"] * (flops_multiattn + flops_ffn)
        flops_final_linear = 2 * config["d_model"] * config["vocab_size"]
        total_seq_flops = (flops_n_layers + flops_final_linear) * config["context_length"] * batch_size # seq = context_length
        print(
            f"For batch size {batch_size},",
            f"for one token, the multiattn FLOPs: {flops_multiattn / 1e9:.2f} GFLOPs, ",
            f"ffn FLOPs: {flops_ffn / 1e9:.2f} GFLOPs, ",
            f"n_layers FLOPs: {flops_n_layers / 1e9:.2f} GFLOPs, ",
            f"final linear FLOPs: {flops_final_linear / 1e9:.2f} GFLOPs, ",
            f"Total seq FLOPs: {total_seq_flops / 1e12:.2f} TFLOPs",
        )
        return total_seq_flops

    steps = 400000 # 400k steps
    mfu = 0.5
    effective_flops_a100 = 19.5 * 1e12 * mfu # 19.5 TFLOPs * mfu
    flops = _flops(xl_large,batch_size=1024) * 3 # forward + backward, optimizier ignore
    total_time_days = (steps * flops) / effective_flops_a100 / 3600 / 24 
    print(
        f"Total training time estimate (days) for batch size 1024 and 400k steps on single A100 GPU: {total_time_days:.2f} days"
    )


if __name__ == "__main__":
    learning_rate_tuning([1e-1, 1e-2, 1e-3])
    adamwAccounting()

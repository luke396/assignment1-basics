"""Blocks implementation for llm."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
from einops import einsum, rearrange
from torch import nn


class Linear(nn.Module):
    """A linear block without bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a linear layer without bias.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            device: Device to create the weight tensor on.
            dtype: Data type for the weight tensor.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        sigma = 2 / (in_features + out_features)
        nn.init.trunc_normal_(  # init with suggested truncated normal
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the linear layer."""
        return einsum(x, self.weight, "batch ... d_in, d_out d_in -> batch ... d_out")


class Embedding(nn.Module):
    """An embedding block."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize an embedding layer.

        Args:
            num_embeddings: Number of embeddings, vocabulary_size.
            embedding_dim: Dimension of each embedding vector, d_model.
            device: Device to create the embedding tensor on.
            dtype: Data type for the embedding tensor.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the embedding layer."""
        return self.weight[x]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize RMSNorm layer.

        Args:
            d_model: Dimension of the model.
            eps: Small constant for numerical stability.
            device: Device to create the weight tensor on.
            dtype: Data type for the weight tensor.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor with same shape as input.

        """
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        inv_rms = torch.rsqrt(
            x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # rms along with d_model
        y = x_fp32 * inv_rms * self.weight.to(torch.float32)
        return y.to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize SiLU feed-forward network.

        Args:
            d_model: Dimension of the model input and output.
            d_ff: Dimension of the feed-forward hidden layer.
            device: Device to create the weight tensors on.
            dtype: Data type for the weight tensors.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w2 = Linear(self.d_ff, self.d_model, **factory_kwargs)
        self.w3 = Linear(self.d_model, self.d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input tensor.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).

        """
        w1_out = self.w1(x)  # pre-cache reduce redundant computation
        silu = w1_out * torch.sigmoid(w1_out)
        return self.w2(silu * self.w3(x))


class SiLU(nn.Module):
    """SiLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize SwiGLU feed-forward network.

        Args:
            d_model: Dimension of the model input and output.
            d_ff: Dimension of the feed-forward hidden layer.
            device: Device to create the weight tensors on.
            dtype: Data type for the weight tensors.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w2 = Linear(self.d_ff, self.d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU transformation to input tensor.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).

        """
        w1_out = self.w1(x)
        silu = w1_out * torch.sigmoid(w1_out)
        return self.w2(silu)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) block.

    This module precomputes and caches rotation matrices for efficient
    application of rotary positional embeddings.

    Buffers:
        inv_freq: Inverse frequencies for rotation computation, shape (d_k // 2,).
        cos_cached: Precomputed cosine values, shape (max_seq_len, d_k // 2).
        sin_cached: Precomputed sine values, shape (max_seq_len, d_k // 2).

    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        """Initialize Rotary Positional Embedding.

        Args:
            theta: Base value for frequency computation.
            d_k: Dimension of key/query vectors (must be even).
            max_seq_len: Maximum sequence length to precompute embeddings for.
            device: Device to create tensors on.

        """
        assert d_k % 2 == 0, "d_k must be even for RoPE."
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin lookup tables up to max_seq_len for efficiency.
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        phi: torch.Tensor = einsum(positions, self.inv_freq, "n, d -> n d")  # type: ignore[arg-type]
        self.register_buffer("cos_cached", torch.cos(phi))
        self.register_buffer("sin_cached", torch.sin(phi))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor.

        Args:
            x: Input tensor of shape (..., d_k).
            token_positions: Token position indices of shape (...).

        Returns:
            Output tensor with rotary embeddings applied, same shape as input.

        """
        # Ensure indices are on the same device and of integer type
        token_positions = token_positions.to(self.cos_cached.device).long()  # type: ignore[union-attr]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        cos_phi = self.cos_cached[token_positions].to(dtype=x.dtype)  # type: ignore[index]
        sin_phi = self.sin_cached[token_positions].to(dtype=x.dtype)  # type: ignore[index]
        y_even = x_even * cos_phi - x_odd * sin_phi
        y_odd = x_even * sin_phi + x_odd * cos_phi
        y = torch.empty_like(x)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_odd
        return y


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute softmax along specified dimension."""
    x = x - x.amax(dim=dim, keepdim=True)  # for numerical stability
    exp_x = torch.exp(x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch_size, ..., seq_len_q, d_k).
        k: Key tensor of shape (batch_size, ..., seq_len_k, d_k).
        v: Value tensor of shape (batch_size, ..., seq_len_k, d_v).
        mask: Optional boolean mask tensor of shape (..., seq_len_q, seq_len_k).
              True indicates positions to keep, False indicates positions to mask out.

    Returns:
        Attention output tensor of shape (batch_size, ..., seq_len_q, d_v).

    """
    # Scaled dot-product for numerical stability
    d_k = k.shape[-1]
    scores = einsum(q, k, "... s_q d_k, ... s_k d_k -> ... s_q s_k") / torch.sqrt(
        torch.tensor(d_k, dtype=q.dtype, device=q.device)
    )

    if mask is not None:
        neg_inf = torch.tensor(
            float("-inf"), dtype=scores.dtype, device=scores.device
        )  # Convert boolean mask to additive mask: True -> 0.0, False -> -inf
        scores = torch.where(mask, scores, neg_inf)

    attention_weights = softmax(scores, dim=-1)
    return einsum(attention_weights, v, "... s_q s_k, ... s_k d_k -> ... s_q d_k")


class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(  # noqa: PLR0913
        self,
        d_model: int,
        num_heads: int,
        seq_len: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize multi-head self-attention.

        Args:
            d_model: Dimension of the model (must be divisible by num_heads).
            num_heads: Number of attention heads.
            seq_len: Optional max sequence length for caching the causal mask.
            rope: Optional RotaryPositionalEmbedding module for RoPE.
            device: Device to create tensors on.
            dtype: Data type for tensors.

        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.seq_len = seq_len
        if self.seq_len:
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(
                        (self.seq_len, self.seq_len), device=device, dtype=torch.bool
                    ),
                    diagonal=0,
                ),
                persistent=False,
            )

        self.register_buffer(
            "k_cache",
            None,
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            None,
            persistent=False,
        )

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(
        self,
        in_features: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            in_features: Input tensor of shape (..., seq_len, d_model).
            token_positions: Optional token position indices of shape (seq_len,).
                Will broadcast automatically to match batch and head dimensions.
            use_cache: If True, use and update the KV cache for incremental decoding.

        Returns:
            Output tensor of shape (..., seq_len, d_model).

        """
        q = self.q_proj(in_features)  # (..., seq_len, d_model)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)

        # Split into multiple heads
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
        q = rearrange(
            q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads
        )
        k = rearrange(
            k, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads
        )
        v = rearrange(
            v, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads
        )

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # This should be after RoPE because cached KV has already been rotated.
        if use_cache:
            if self.k_cache is not None and self.v_cache is not None:
                k = torch.cat([self.k_cache, k], dim=-2)
                v = torch.cat([self.v_cache, v], dim=-2)
            if self.seq_len and k.shape[-2] > self.seq_len:
                msg = (
                    f"KV cache length {k.shape[-2]} exceeds "
                    f"context length {self.seq_len}"
                )
                raise ValueError(msg)
            self.k_cache = k.detach()
            self.v_cache = v.detach()

        # True for positions to keep, False to mask
        q_len = q.shape[-2]
        k_len = k.shape[-2]

        if not self.seq_len:
            mask = torch.tril(
                torch.ones((q_len, k_len), device=in_features.device, dtype=torch.bool),
                diagonal=k_len - q_len,
            )
        else:
            assert isinstance(self.mask, torch.Tensor)
            mask = self.mask[k_len - q_len : k_len, :k_len]

        attention_output = scaled_dot_product_attention(
            q, k, v, mask=mask
        )  # (..., num_heads, seq_len, d_k)

        # Merge heads
        # (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        attention_output = rearrange(
            attention_output, "... heads seq d_k -> ... seq (heads d_k)"
        )
        return self.output_proj(attention_output)

    def clear_kv_cache(self) -> None:
        """Clear the key and value caches."""
        self.k_cache = None
        self.v_cache = None


class TransformerBlock(nn.Module):
    """Transformer block with configurable normalization strategy.

    Buffers:
        _pos_cache: Cached position indices for sequence positions,
                   shape (context_length,). Only allocated if context_length
                   is provided during initialization.
                   Not persisted in model checkpoints.

    """

    def __init__(  # noqa: PLR0913
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        context_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_strategy: Literal["pre", "post", "none"] = "pre",
        ffn_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize Transformer block.

        Args:
            d_model: Dimension of the model.
            num_heads: Number of attention heads.
            d_ff: Dimension of the feed-forward network.
            rope: Optional RotaryPositionalEmbedding module for RoPE.
            context_length: Optional maximum context length for
                pre-allocating position cache.
            device: Device to create tensors on.
            dtype: Data type for tensors.
            norm_strategy: Where to place RMSNorm layers:
                "pre" for pre-norm (default), "post" for post-norm,
                "none" to disable normalization.
            ffn_factory: Optional factory for creating the feed-forward
                network; it must accept (d_model, d_ff, device, dtype).
                Defaults to SwiGLU.

        """
        super().__init__()
        if norm_strategy not in {"pre", "post", "none"}:
            msg = f"Unknown norm_strategy: {norm_strategy}"
            raise ValueError(msg)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm_strategy = norm_strategy
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            seq_len=context_length,
            rope=rope,
            **factory_kwargs,
        )
        ffn_ctor = ffn_factory if ffn_factory is not None else SwiGLU
        self.ffn = ffn_ctor(d_model=d_model, d_ff=d_ff, **factory_kwargs)
        self.ln1 = (
            RMSNorm(d_model, **factory_kwargs) if norm_strategy != "none" else None
        )
        self.ln2 = (
            RMSNorm(d_model, **factory_kwargs) if norm_strategy != "none" else None
        )

        # Pre-allocate position cache if context_length is provided
        if context_length:
            positions: torch.Tensor = torch.arange(
                context_length, device=device, dtype=torch.long
            )
            self.register_buffer("_pos_cache", positions, persistent=False)
            self._pos_cache_len = context_length
        else:
            # Cache for position indices
            self.register_buffer("_pos_cache", None, persistent=False)
            self._pos_cache_len = 0

    def _get_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Get or create cached position indices for input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).

        Returns:
            Position indices of shape (seq_len,).
            Will broadcast automatically to match batch dimensions when used.

        """
        seq_len = x.shape[-2]

        if self._pos_cache is None or self._pos_cache_len < seq_len:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            self.register_buffer("_pos_cache", positions, persistent=False)
            self._pos_cache_len = seq_len

        # Use cached positions (slice to current seq_len)
        # Return 1D tensor that will broadcast automatically
        assert self._pos_cache is not None
        return self._pos_cache[:seq_len]  # type: ignore[index]

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Apply Transformer block.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).
            token_positions: Optional tensor of position indices of shape
                (seq_len,). If None, positions are obtained from the internal
                position cache via :meth:`_get_positions`.
            use_cache: If True, use and update the KV cache for incremental
                decoding.

        Returns:
            Output tensor of shape (..., seq_len, d_model).

        """
        if token_positions is None:
            token_positions = self._get_positions(x)

        if self.norm_strategy == "pre":
            attn_in = self.ln1(x) if self.ln1 is not None else x
            x1 = x + self.attn(attn_in, token_positions, use_cache=use_cache)
            ffn_in = self.ln2(x1) if self.ln2 is not None else x1
            return x1 + self.ffn(ffn_in)

        if self.norm_strategy == "post":
            x1 = x + self.attn(x, token_positions, use_cache=use_cache)
            x1 = self.ln1(x1) if self.ln1 is not None else x1
            x2 = x1 + self.ffn(x1)
            return self.ln2(x2) if self.ln2 is not None else x2

        # norm_strategy == "none"  # noqa: ERA001
        x1 = x + self.attn(x, token_positions, use_cache=use_cache)
        return x1 + self.ffn(x1)


class TransformerLM(nn.Module):
    """Transformer Language Model."""

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        n_layers: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_strategy: Literal["pre", "post", "none"] = "pre",
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
    ) -> None:
        """Initialize Transformer Language Model."""
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        ffn_factory = SiLU if ffn_type == "silu" else None
        add_final_norm = norm_strategy != "none"
        block_factory = partial(
            TransformerBlock,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope=rope,
            context_length=context_length,
            device=device,
            dtype=dtype,
            norm_strategy=norm_strategy,
            ffn_factory=ffn_factory,
        )
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList([block_factory() for _ in range(n_layers)])
        self.ln_final = RMSNorm(d_model, **factory_kwargs) if add_final_norm else None
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self,
        in_indices: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Apply Transformer Language Model.

        Args:
            in_indices: Input token indices of shape (..., seq_len).
            token_positions: Optional token position indices of shape
                (..., seq_len). If None, positions are inferred from the
                internal position cache.
            use_cache: If True, use and update the KV cache for incremental
                decoding.

        Returns:
            Logits over the vocabulary of shape (..., seq_len, vocab_size).

        """
        x = self.token_embeddings(in_indices)  # (..., seq_len, d_model)
        for block in self.layers:
            x = block(
                x, token_positions, use_cache=use_cache
            )  # (..., seq_len, d_model)
        if self.ln_final is not None:
            x = self.ln_final(x)  # (..., seq_len, d_model)
        return self.lm_head(x)  # (..., seq_len, vocab_size)

    def clear_kv_cache(self) -> None:
        """Clear KV cache for all attention layers."""
        for block in self.layers:
            if isinstance(block, TransformerBlock):
                block.attn.clear_kv_cache()

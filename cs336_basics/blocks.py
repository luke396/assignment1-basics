"""Blocks implementation for llm."""

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

        inv_freq: torch.Tensor = 1.0 / (
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

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize multi-head self-attention.

        Args:
            d_model: Dimension of the model (must be divisible by num_heads).
            num_heads: Number of attention heads.
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

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(
        self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            in_features: Input tensor of shape (..., seq_len, d_model).
            token_positions: Optional token position indices of shape (seq_len,).
                Will broadcast automatically to match batch and head dimensions.

        Returns:
            Output tensor of shape (..., seq_len, d_model).

        """
        seq_len = in_features.shape[-2]

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

        # True for positions to keep, False to mask
        mask = torch.tril(
            torch.ones((seq_len, seq_len), device=in_features.device, dtype=torch.bool),
            diagonal=0,
        )

        attention_output = scaled_dot_product_attention(
            q, k, v, mask=mask
        )  # (..., num_heads, seq_len, d_k)

        # Merge heads
        # (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        attention_output = rearrange(
            attention_output, "... heads seq d_k -> ... seq (heads d_k)"
        )
        return self.output_proj(attention_output)


class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention and feed-forward network.

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

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            **factory_kwargs,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, **factory_kwargs)
        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, **factory_kwargs)

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
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply Transformer block.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).
            token_positions: Optional token position indices of shape (seq_len,).
                If None, defaults to [0, 1, 2, ..., seq_len-1].
                Will broadcast automatically to match batch dimensions.

        Returns:
            Output tensor of shape (..., seq_len, d_model).

        """
        if token_positions is None:
            token_positions = self._get_positions(x)

        # First residual connection: attention
        x1 = x + self.attn(self.ln1(x), token_positions)
        # Second residual connection: feed-forward
        return x1 + self.ffn(self.ln2(x1))


class TransformerBlockNoRMSNorm(nn.Module):
    """Transformer block without any normalization layers."""

    def __init__(  # noqa: PLR0913
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        context_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize Transformer block without RMSNorm."""
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            **factory_kwargs,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, **factory_kwargs)

        if context_length:
            positions: torch.Tensor = torch.arange(
                context_length, device=device, dtype=torch.long
            )
            self.register_buffer("_pos_cache", positions, persistent=False)
            self._pos_cache_len = context_length
        else:
            self.register_buffer("_pos_cache", None, persistent=False)
            self._pos_cache_len = 0

    def _get_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Get or create cached position indices for input tensor."""
        seq_len = x.shape[-2]

        if self._pos_cache is None or self._pos_cache_len < seq_len:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            self.register_buffer("_pos_cache", positions, persistent=False)
            self._pos_cache_len = seq_len

        assert self._pos_cache is not None
        return self._pos_cache[:seq_len]  # type: ignore[index]

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply Transformer block without normalization."""
        if token_positions is None:
            token_positions = self._get_positions(x)

        x1 = x + self.attn(x, token_positions)
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
    ) -> None:
        """Initialize Transformer Language Model.

        Args:
            vocab_size: Number of embeddings in vocabulary.
            d_model: Model dimension (also used for token embeddings).
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            context_length: Maximum context length for position caching.
            n_layers: Number of transformer blocks.
            rope: Optional RotaryPositionalEmbedding module.
            device: Device to create tensors on.
            dtype: Data type for tensors.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope,
                    context_length,
                    **factory_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """Apply Transformer Language Model.

        Args:
            in_indices: Input token indices of shape (..., seq_len).

        Returns:
            Output logits of shape (..., seq_len, num_embeddings).

        """
        x = self.token_embeddings(in_indices)  # (..., seq_len, d_model)
        for block in self.layers:
            x = block(x)  # (..., seq_len, d_model)
        x = self.ln_final(x)  # (..., seq_len, d_model)
        return self.lm_head(x)  # (..., seq_len, vocab_size)


class TransformerLMNoRMSNorm(nn.Module):
    """Transformer Language Model that omits all RMSNorm layers."""

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
    ) -> None:
        """Initialize a norm-free Transformer Language Model."""
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList(
            [
                TransformerBlockNoRMSNorm(
                    d_model,
                    num_heads,
                    d_ff,
                    rope,
                    context_length,
                    **factory_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """Apply the norm-free Transformer Language Model."""
        x = self.token_embeddings(in_indices)  # (..., seq_len, d_model)
        for block in self.layers:
            x = block(x)  # (..., seq_len, d_model)
        return self.lm_head(x)  # (..., seq_len, vocab_size)

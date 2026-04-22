import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kv_cache import KVCache
from src.models.rmsnorm import RMSNorm
from src.models.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        rope_base: float,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.dropout = float(dropout)

        if self.d_model % self.n_heads != 0:
            raise ValueError("Model dimension must be divisible by number of heads.")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("Number of heads must be divisible by number of KV heads.")

        self.head_dim = self.d_model // self.n_heads

        if self.head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for rotary embeddings.")

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)
        self.rope = RotaryEmbedding(head_dim=self.head_dim, base=float(rope_base))

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        repeat = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _build_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        key_mask = attention_mask.to(torch.bool)[:, None, None, :]
        if query_length == key_length:
            causal = torch.tril(torch.ones((query_length, key_length), dtype=torch.bool, device=device))
            return key_mask & causal.view(1, 1, query_length, key_length)

        return key_mask.expand(-1, 1, query_length, -1)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_index: int | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, query_length, _ = x.shape

        q = self.q_proj(x).view(batch_size, query_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, query_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, query_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope.apply(q=q, k=k, position_ids=position_ids)

        if kv_cache is not None and layer_index is not None:
            cached_k, cached_v = kv_cache.get(layer_index)
            if cached_k is not None and cached_v is not None:
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            if use_cache:
                kv_cache.set(layer_index, k, v)

        key_length = int(k.shape[2])

        k = self._expand_kv(k)
        v = self._expand_kv(v)

        attn_mask = self._build_attention_mask(
            attention_mask=attention_mask,
            query_length=query_length,
            key_length=key_length,
            device=x.device,
        )

        use_causal = attn_mask is None and query_length == key_length

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, query_length, self.d_model)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden_dim = int(d_model) * int(ff_mult)
        self.gate_proj = nn.Linear(int(d_model), hidden_dim, bias=False)
        self.up_proj = nn.Linear(int(d_model), hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, int(d_model), bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class TokenDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ff_mult: int,
        dropout: float,
        rope_base: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(int(d_model))
        self.mlp_norm = RMSNorm(int(d_model))
        self.attn = CausalSelfAttention(
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads),
            dropout=float(dropout),
            rope_base=float(rope_base),
        )
        self.mlp = SwiGLU(
            d_model=int(d_model),
            ff_mult=int(ff_mult),
            dropout=float(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_index: int | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_index=layer_index,
            use_cache=use_cache,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x
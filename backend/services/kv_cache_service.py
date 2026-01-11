"""KV Cache comparison service for Chapter 8 (attention memory efficiency)."""

from __future__ import annotations

from typing import Any


def _format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b} B"


class KVCacheService:
    """Service for calculating KV cache memory usage across attention variants."""

    def get_comparison(
        self,
        context_length: int = 4096,
        n_layers: int = 32,
        d_model: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int | None = None,
        gqa_groups: int = 8,
        mla_latent_dim: int = 512,
        bytes_per_param: int = 2,
    ) -> dict[str, Any]:
        """Calculate KV cache memory usage for different attention variants.

        Compares:
        - MHA: Multi-Head Attention (standard)
        - MQA: Multi-Query Attention (single K/V head)
        - GQA: Grouped-Query Attention (Llama 3 style)
        - MLA: Multi-Head Latent Attention (DeepSeek R1 style)

        Args:
            context_length: Sequence length (tokens in context)
            n_layers: Number of transformer layers
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_kv_heads: Number of KV heads (for GQA, defaults to n_heads // gqa_groups)
            gqa_groups: Number of query heads per KV head group
            mla_latent_dim: Latent dimension for MLA compression
            bytes_per_param: Bytes per parameter (2 for fp16, 4 for fp32)

        Returns:
            Dictionary with memory comparison data
        """
        d_head = d_model // n_heads

        # MHA: Standard Multi-Head Attention
        # Stores K and V for each layer, each head
        # Shape per layer: (context_length, n_heads, d_head) for K and V
        mha_per_layer = 2 * context_length * n_heads * d_head * bytes_per_param
        mha_total = mha_per_layer * n_layers

        # MQA: Multi-Query Attention
        # Uses 1 shared K/V head for all query heads
        # Shape per layer: (context_length, 1, d_head) for K and V
        mqa_per_layer = 2 * context_length * 1 * d_head * bytes_per_param
        mqa_total = mqa_per_layer * n_layers

        # GQA: Grouped-Query Attention (Llama 3)
        # Groups of query heads share K/V heads
        actual_kv_heads = n_kv_heads if n_kv_heads else n_heads // gqa_groups
        gqa_per_layer = 2 * context_length * actual_kv_heads * d_head * bytes_per_param
        gqa_total = gqa_per_layer * n_layers

        # MLA: Multi-Head Latent Attention (DeepSeek R1)
        # Compresses K/V into a low-rank latent space
        # Stores compressed latent: (context_length, latent_dim) per layer
        mla_per_layer = context_length * mla_latent_dim * bytes_per_param
        mla_total = mla_per_layer * n_layers

        architectures = [
            {
                "name": "MHA",
                "full_name": "Multi-Head Attention",
                "description": "Standard attention with separate K/V per head",
                "memory_bytes": mha_total,
                "memory_formatted": _format_bytes(mha_total),
                "ratio_to_mha": 1.0,
                "kv_heads": n_heads,
            },
            {
                "name": "MQA",
                "full_name": "Multi-Query Attention",
                "description": "Single shared K/V head for all queries",
                "memory_bytes": mqa_total,
                "memory_formatted": _format_bytes(mqa_total),
                "ratio_to_mha": round(mqa_total / mha_total, 4),
                "kv_heads": 1,
            },
            {
                "name": "GQA",
                "full_name": "Grouped-Query Attention",
                "description": f"K/V heads shared within groups of {gqa_groups}",
                "memory_bytes": gqa_total,
                "memory_formatted": _format_bytes(gqa_total),
                "ratio_to_mha": round(gqa_total / mha_total, 4),
                "kv_heads": actual_kv_heads,
            },
            {
                "name": "MLA",
                "full_name": "Multi-Head Latent Attention",
                "description": f"Compressed to {mla_latent_dim}-dim latent space",
                "memory_bytes": mla_total,
                "memory_formatted": _format_bytes(mla_total),
                "ratio_to_mha": round(mla_total / mha_total, 4),
                "latent_dim": mla_latent_dim,
            },
        ]

        # Calculate savings
        mha_to_mla_ratio = mha_total / mla_total if mla_total > 0 else 0

        return {
            "config": {
                "context_length": context_length,
                "n_layers": n_layers,
                "d_model": d_model,
                "n_heads": n_heads,
                "d_head": d_head,
                "gqa_groups": gqa_groups,
                "mla_latent_dim": mla_latent_dim,
                "bytes_per_param": bytes_per_param,
            },
            "architectures": architectures,
            "mha_to_mla_savings": round(mha_to_mla_ratio, 1),
            "explanation": (
                f"KV Cache comparison for {context_length:,} tokens across {n_layers} layers. "
                f"MHA (standard) uses {_format_bytes(mha_total)}, while DeepSeek's MLA uses only "
                f"{_format_bytes(mla_total)} — a {mha_to_mla_ratio:.1f}x reduction. "
                "This enables much longer context windows without memory explosion."
            ),
        }

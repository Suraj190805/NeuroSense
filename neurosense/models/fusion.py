"""NeuroSense Cross-Modal Attention Fusion.

Implements the cross-modal fusion module specified in PRD
Section 4.2.4 (Phase 4):
- Multi-head cross-attention: Query=MRI embedding, Key/Value=Clinical
- Pre-norm transformer block with residual connections
- FFN: Linear(256→1024) → GELU → Linear(1024→256)
- Output: fused 256-dim representation

Architecture::

    img_emb [B, 256]    clin_emb [B, 256]
        ↓                     ↓
    unsqueeze [B,1,256]  unsqueeze [B,1,256]
        ↓                     ↓
        └──── CrossAttention ─┘
              Q=img  K,V=clin
                  ↓
        LayerNorm → MHA → + (residual)
                  ↓
        LayerNorm → FFN → + (residual)
                  ↓
              [B, 1, 256]
                  ↓
              squeeze → [B, 256]

Usage:
    from neurosense.models.fusion import CrossModalFusion

    fusion = CrossModalFusion(embed_dim=256, num_heads=8)
    img_emb = torch.randn(4, 256)   # from MRIEncoder
    clin_emb = torch.randn(4, 256)  # from ClinicalEncoder
    fused = fusion(img_emb, clin_emb)  # [4, 256]
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network for transformer blocks.

    Two-layer MLP with GELU activation, as specified in PRD 4.2.4:
    ``Linear(d, d_ff) → GELU → Dropout → Linear(d_ff, d) → Dropout``

    Args:
        embed_dim: Input and output dimension (default: 256).
        ffn_hidden_dim: Hidden layer dimension (default: 1024).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> ffn = FeedForwardNetwork(256, 1024)
        >>> x = torch.randn(4, 1, 256)
        >>> out = ffn(x)  # [4, 1, 256]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        ffn_hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

        # Kaiming initialisation
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="linear"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape ``[B, T, embed_dim]``.

        Returns:
            Output tensor of same shape.
        """
        return self.net(x)


class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion for MRI and clinical embeddings.

    Implements a pre-norm transformer cross-attention block that
    fuses imaging and clinical modalities (PRD Section 4.2.4).
    The MRI embedding attends to the clinical embedding via
    multi-head attention (Query=MRI, Key/Value=Clinical).

    Pre-norm architecture (more stable training than post-norm)::

        x_norm = LayerNorm(x)
        attn_out = MultiheadAttention(Q=x_norm, K=clin_norm, V=clin_norm)
        x = x + attn_out          # Residual connection 1
        x_norm = LayerNorm(x)
        ffn_out = FFN(x_norm)
        x = x + ffn_out           # Residual connection 2

    Args:
        embed_dim: Embedding dimension for both modalities
            (default: 256, matching encoder outputs).
        num_heads: Number of attention heads (default: 8 per PRD).
            Must evenly divide embed_dim (256/8 = 32 per head).
        ffn_hidden_dim: FFN hidden dimension (default: 1024,
            giving 4× expansion as in standard transformers).
        dropout: Dropout for attention weights and FFN
            (default: 0.1).
        pre_norm: If True, apply LayerNorm before attention/FFN
            (default: True per PRD). If False, apply post-norm.

    Attributes:
        norm1: LayerNorm before cross-attention.
        norm2: LayerNorm before FFN.
        norm_kv: LayerNorm for clinical (key/value) input.
        cross_attention: Multi-head cross-attention module.
        ffn: Position-wise feed-forward network.

    Example:
        >>> fusion = CrossModalFusion(embed_dim=256, num_heads=8)
        >>> img = torch.randn(4, 256)
        >>> clin = torch.randn(4, 256)
        >>> fused = fusion(img, clin)  # [4, 256]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_hidden_dim: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pre_norm = pre_norm

        # ─── Layer Norms ───
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # ─── Cross-Attention (PRD 4.2.4) ───
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ─── Feed-Forward Network ───
        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

        # ─── Dropout for residual connections ───
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        # Log architecture
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CrossModalFusion initialised: "
            "embed=%d, heads=%d, ffn=%d, pre_norm=%s | %dK params",
            embed_dim,
            num_heads,
            ffn_hidden_dim,
            pre_norm,
            total_params // 1_000,
        )

    def forward(
        self,
        img_emb: torch.Tensor,
        clin_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse MRI and clinical embeddings via cross-attention.

        Args:
            img_emb: MRI embedding of shape ``[B, embed_dim]`` or
                ``[B, 1, embed_dim]``. Used as Query.
            clin_emb: Clinical embedding of shape ``[B, embed_dim]``
                or ``[B, 1, embed_dim]``. Used as Key and Value.

        Returns:
            Fused embedding of shape ``[B, embed_dim]``.
        """
        # Ensure 3D for attention: [B, D] → [B, 1, D]
        if img_emb.ndim == 2:
            img_emb = img_emb.unsqueeze(1)
        if clin_emb.ndim == 2:
            clin_emb = clin_emb.unsqueeze(1)

        if self.pre_norm:
            fused = self._forward_pre_norm(img_emb, clin_emb)
        else:
            fused = self._forward_post_norm(img_emb, clin_emb)

        # Squeeze back to [B, D]
        return fused.squeeze(1)

    def _forward_pre_norm(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-norm transformer block (default, more stable).

        Args:
            query: Query tensor ``[B, 1, D]`` (MRI embedding).
            key_value: Key/Value tensor ``[B, 1, D]`` (clinical).

        Returns:
            Fused tensor ``[B, 1, D]``.
        """
        # Cross-attention with pre-norm
        q_norm = self.norm1(query)
        kv_norm = self.norm_kv(key_value)

        attn_out, _attn_weights = self.cross_attention(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
        )
        # Residual connection 1
        x = query + self.dropout1(attn_out)

        # FFN with pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        # Residual connection 2
        x = x + self.dropout2(ffn_out)

        return x

    def _forward_post_norm(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        """Post-norm transformer block (alternative).

        Args:
            query: Query tensor ``[B, 1, D]``.
            key_value: Key/Value tensor ``[B, 1, D]``.

        Returns:
            Fused tensor ``[B, 1, D]``.
        """
        # Cross-attention
        attn_out, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
        )
        x = self.norm1(query + self.dropout1(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x

    def get_attention_weights(
        self,
        img_emb: torch.Tensor,
        clin_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Extract cross-attention weights for explainability.

        Returns the attention weight matrix showing how much the
        MRI representation attends to clinical features. Useful
        for interpreting which clinical signals influence the
        fused representation.

        Args:
            img_emb: MRI embedding ``[B, embed_dim]``.
            clin_emb: Clinical embedding ``[B, embed_dim]``.

        Returns:
            Attention weights ``[B, num_heads, 1, 1]``.
        """
        if img_emb.ndim == 2:
            img_emb = img_emb.unsqueeze(1)
        if clin_emb.ndim == 2:
            clin_emb = clin_emb.unsqueeze(1)

        q = self.norm1(img_emb) if self.pre_norm else img_emb
        kv = self.norm_kv(clin_emb) if self.pre_norm else clin_emb

        _, attn_weights = self.cross_attention(
            query=q,
            key=kv,
            value=kv,
            average_attn_weights=False,
        )

        return attn_weights


class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion (ablation baseline).

    Concatenates MRI and clinical embeddings and projects back
    to the embedding dimension. Used as a baseline in ablation
    studies (PRD Section 11) to compare against cross-attention.

    Architecture::

        [img_emb; clin_emb]  →  Linear(512, 256) → ReLU → LayerNorm
              [B, 512]                     [B, 256]

    Args:
        embed_dim: Input embedding dimension per modality
            (default: 256).

    Example:
        >>> fusion = ConcatenationFusion(embed_dim=256)
        >>> fused = fusion(img_emb, clin_emb)  # [B, 256]
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dim),
        )

        logger.info(
            "ConcatenationFusion initialised: 2×%d → %d",
            embed_dim,
            embed_dim,
        )

    def forward(
        self,
        img_emb: torch.Tensor,
        clin_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse via concatenation and projection.

        Args:
            img_emb: MRI embedding ``[B, embed_dim]``.
            clin_emb: Clinical embedding ``[B, embed_dim]``.

        Returns:
            Fused embedding ``[B, embed_dim]``.
        """
        if img_emb.ndim == 3:
            img_emb = img_emb.squeeze(1)
        if clin_emb.ndim == 3:
            clin_emb = clin_emb.squeeze(1)

        concat = torch.cat([img_emb, clin_emb], dim=-1)
        return self.projection(concat)

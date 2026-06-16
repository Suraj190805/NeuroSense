"""NeuroSense Bi-LSTM Clinical Encoder.

Implements the clinical feature encoder specified in PRD
Section 4.2.3 (Phase 3):
- 2-layer bidirectional LSTM for temporal clinical sequences
- Handles variable-length visit sequences via pack/pad
- Projects 512-dim Bi-LSTM output to 256-dim embedding
- Supports both single-visit and multi-visit inputs

Architecture::

    Input: [B, T, 5]  (T visits × 5 clinical features)
        ↓
    2-layer Bi-LSTM (hidden=256, dropout=0.3)
        ↓
    [B, 512] (last valid timestep, forward + backward concat)
        ↓
    Linear(512, 256) → ReLU → LayerNorm
        ↓
    [B, 256] (embedding)

Clinical feature vector (normalised to [0,1]):
    [cag_repeat, uhdrs_motor, uhdrs_cognitive, tfc, age]

Usage:
    from neurosense.models.clinical_encoder import ClinicalEncoder

    # Single-visit (most common in inference)
    encoder = ClinicalEncoder()
    clinical = torch.randn(4, 1, 5)  # 4 subjects, 1 visit, 5 features
    lengths = torch.ones(4, dtype=torch.long)
    embedding = encoder(clinical, lengths)  # [4, 256]

    # Multi-visit longitudinal sequence
    clinical_seq = torch.randn(4, 6, 5)  # 4 subjects, up to 6 visits
    lengths = torch.tensor([6, 4, 3, 6])  # actual visit counts
    embedding = encoder(clinical_seq, lengths)  # [4, 256]
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


class ClinicalEncoder(nn.Module):
    """Bidirectional LSTM encoder for longitudinal clinical features.

    Encodes variable-length sequences of clinical visit data into
    a fixed-size 256-dim embedding vector for downstream fusion
    with the MRI encoder (PRD Section 4.2.3).

    The encoder uses ``pack_padded_sequence`` to efficiently handle
    variable-length visit sequences without computing over padding
    tokens. The final hidden state from both LSTM directions is
    concatenated and projected to the embedding space.

    Args:
        input_features: Number of clinical features per visit
            (default: 5 — cag_repeat, uhdrs_motor, uhdrs_cognitive,
            tfc, age).
        hidden_size: LSTM hidden state dimension per direction
            (default: 256 per PRD). Total Bi-LSTM output is
            ``2 * hidden_size = 512``.
        num_layers: Number of stacked LSTM layers (default: 2
            per PRD Section 4.2.3).
        dropout: Dropout probability between LSTM layers
            (default: 0.3 per PRD). Applied between layers only
            when ``num_layers > 1``.
        bidirectional: Whether to use bidirectional LSTM
            (default: True per PRD).
        embedding_dim: Output embedding dimension (default: 256
            to match MRIEncoder output for fusion).

    Attributes:
        lstm: Bi-LSTM module (2-layer, hidden=256, bidirectional).
        projection: Linear(512, 256) → ReLU → LayerNorm.
        output_dropout: Dropout applied to LSTM output before
            projection.

    Example:
        >>> encoder = ClinicalEncoder(
        ...     input_features=5,
        ...     hidden_size=256,
        ...     num_layers=2,
        ...     dropout=0.3,
        ... )
        >>> # Single visit per subject
        >>> x = torch.randn(4, 1, 5)
        >>> lengths = torch.ones(4, dtype=torch.long)
        >>> emb = encoder(x, lengths)  # [4, 256]
        >>>
        >>> # Multi-visit longitudinal data
        >>> x = torch.randn(4, 6, 5)  # padded to 6 visits
        >>> lengths = torch.tensor([6, 4, 3, 2])
        >>> emb = encoder(x, lengths)  # [4, 256]
    """

    def __init__(
        self,
        input_features: int = 5,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim

        # Bi-LSTM output dimension: 2 * hidden_size if bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_dim = hidden_size * self.num_directions

        # ─── Input feature projection ───
        # Optional input projection to increase feature dimensionality
        # before LSTM. This helps the LSTM learn richer representations
        # from the 5-dim clinical input.
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_size),
        )

        # ─── Bi-LSTM (PRD 4.2.3) ───
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # After input projection
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ─── Output dropout ───
        self.output_dropout = nn.Dropout(p=dropout)

        # ─── Projection head (PRD 4.2.3) ───
        # Linear(512, 256) → ReLU → LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(self.lstm_output_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
        )

        # Initialise weights
        self._init_weights()

        # Log architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "ClinicalEncoder initialised: "
            "input=%d → LSTM(%d×%d, h=%d, bi=%s, drop=%.1f) "
            "→ proj(%d→%d) | %dK params",
            input_features,
            num_layers,
            self.num_directions,
            hidden_size,
            bidirectional,
            dropout,
            self.lstm_output_dim,
            embedding_dim,
            total_params // 1_000,
        )

    def _init_weights(self) -> None:
        """Initialise LSTM and projection weights.

        Uses orthogonal initialisation for LSTM recurrent weights
        (helps with gradient flow through time) and Kaiming normal
        for linear layers.
        """
        # LSTM weight initialisation
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                # Hidden-hidden weights: orthogonal (better for RNNs)
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                # Bias: zero, except forget gate bias set to 1.0
                # to encourage remembering at initialisation
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1.0
                # For LSTM, biases are [input, forget, cell, output]
                # each of size hidden_size
                n = param.size(0)
                forget_start = n // 4
                forget_end = n // 2
                param.data[forget_start:forget_end].fill_(1.0)

        # Projection weights: Kaiming normal
        for module in self.input_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode clinical visit sequence to 256-dim embedding.

        Handles variable-length visit sequences using
        ``pack_padded_sequence`` for efficient LSTM computation.
        Extracts the hidden state at the last valid timestep for
        each sequence in the batch.

        Args:
            x: Clinical feature tensor of shape
                ``[B, T, input_features]`` where B is batch size,
                T is the maximum number of visits (padded), and
                input_features=5. Zero-padded beyond actual visits.
            lengths: Integer tensor of shape ``[B]`` containing
                the actual number of visits per subject. Must
                satisfy ``1 <= lengths[i] <= T`` for all i.

        Returns:
            Embedding tensor of shape ``[B, embedding_dim]``
            (default: ``[B, 256]``).

        Raises:
            ValueError: If input tensor dimensions don't match
                expected shape.
        """
        batch_size, max_seq_len, n_features = x.shape

        if n_features != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} input features, "
                f"got {n_features}"
            )

        # Clamp lengths to valid range
        lengths = lengths.clamp(min=1, max=max_seq_len)

        # ─── Input projection ───
        # [B, T, 5] → [B, T, hidden_size]
        x = self.input_projection(x)

        # ─── Pack padded sequences ───
        # Sort by length (descending) for pack_padded_sequence
        sorted_lengths, sort_indices = lengths.sort(descending=True)
        sorted_x = x[sort_indices]

        packed = pack_padded_sequence(
            sorted_x,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        # ─── Bi-LSTM forward ───
        # packed_output contains output at each timestep
        # h_n has shape [num_layers * num_directions, B, hidden_size]
        packed_output, (h_n, _c_n) = self.lstm(packed)

        # ─── Extract last valid hidden state ───
        # For bidirectional LSTM, we want the final hidden states
        # from both directions at the last layer:
        #   - Forward:  h_n[-2] (last layer, forward)
        #   - Backward: h_n[-1] (last layer, backward)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # Each is [B, hidden_size], concat → [B, 2*hidden_size]
            h_forward = h_n[-2]   # Last layer, forward direction
            h_backward = h_n[-1]  # Last layer, backward direction
            lstm_out = torch.cat([h_forward, h_backward], dim=-1)
        else:
            # Unidirectional: just use last layer's hidden state
            lstm_out = h_n[-1]  # [B, hidden_size]

        # ─── Unsort to restore original batch order ───
        _, unsort_indices = sort_indices.sort()
        lstm_out = lstm_out[unsort_indices]

        # ─── Dropout + Projection ───
        lstm_out = self.output_dropout(lstm_out)

        # [B, lstm_output_dim] → [B, embedding_dim]
        embedding = self.projection(lstm_out)

        return embedding

    def forward_single_visit(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method for single-visit (non-longitudinal) input.

        Many inference scenarios have only one clinical visit per
        subject. This method handles the common case without
        requiring the caller to construct lengths.

        Args:
            x: Clinical feature tensor of shape ``[B, 5]`` or
                ``[B, 1, 5]`` (single visit per subject).

        Returns:
            Embedding tensor of shape ``[B, embedding_dim]``.

        Example:
            >>> encoder = ClinicalEncoder()
            >>> clinical = torch.randn(4, 5)
            >>> emb = encoder.forward_single_visit(clinical)  # [4, 256]
        """
        if x.ndim == 2:
            # [B, 5] → [B, 1, 5]
            x = x.unsqueeze(1)

        batch_size = x.size(0)
        lengths = torch.ones(batch_size, dtype=torch.long, device=x.device)

        return self.forward(x, lengths)

    def get_feature_dim(self) -> int:
        """Return the output embedding dimension.

        Returns:
            Integer embedding dimension (256 by default).
        """
        return self.embedding_dim

    def get_lstm_output_dim(self) -> int:
        """Return the raw LSTM output dimension (before projection).

        Returns:
            Integer LSTM output dim (512 for bidirectional with
            hidden_size=256).
        """
        return self.lstm_output_dim


# ═════════════════════════════════════════════════════════════════
#  Temporal Attention Pooling (optional enhancement)
# ═════════════════════════════════════════════════════════════════


class TemporalAttentionPooling(nn.Module):
    """Attention-based pooling over LSTM output timesteps.

    Instead of using only the final hidden state, this module
    computes a weighted average of all LSTM timestep outputs,
    where weights are learned via a small attention network.
    This can capture important clinical patterns at any visit,
    not just the last one.

    This is an optional enhancement over the basic ``ClinicalEncoder``
    and can be enabled for ablation studies.

    Args:
        input_dim: Dimension of LSTM output at each timestep
            (default: 512 for bidirectional LSTM with hidden=256).
        attention_dim: Hidden dimension of the attention network
            (default: 128).

    Example:
        >>> pool = TemporalAttentionPooling(input_dim=512)
        >>> lstm_outputs = torch.randn(4, 6, 512)  # B, T, D
        >>> mask = torch.tensor([[1,1,1,1,1,1],
        ...                      [1,1,1,1,0,0],
        ...                      [1,1,1,0,0,0],
        ...                      [1,1,0,0,0,0]], dtype=torch.bool)
        >>> pooled = pool(lstm_outputs, mask)  # [4, 512]
    """

    def __init__(
        self,
        input_dim: int = 512,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

        logger.info(
            "TemporalAttentionPooling initialised: "
            "input=%d, attention=%d",
            input_dim,
            attention_dim,
        )

    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention-weighted pooling over timesteps.

        Args:
            lstm_output: LSTM output tensor of shape ``[B, T, D]``.
            mask: Boolean mask of shape ``[B, T]`` where True
                indicates valid timesteps. If None, all timesteps
                are considered valid.

        Returns:
            Pooled tensor of shape ``[B, D]``.
        """
        # Compute attention scores: [B, T, D] → [B, T, 1]
        scores = self.attention(lstm_output)

        # Mask invalid timesteps
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.unsqueeze(-1), float("-inf")
            )

        # Softmax over time dimension: [B, T, 1]
        weights = torch.softmax(scores, dim=1)

        # Weighted sum: [B, T, D] * [B, T, 1] → sum → [B, D]
        pooled = (lstm_output * weights).sum(dim=1)

        return pooled


# ═════════════════════════════════════════════════════════════════
#  ClinicalEncoder with Attention Pooling Variant
# ═════════════════════════════════════════════════════════════════


class ClinicalEncoderWithAttention(ClinicalEncoder):
    """ClinicalEncoder variant using temporal attention pooling.

    Extends the base ``ClinicalEncoder`` by replacing the final
    hidden state extraction with attention-weighted pooling over
    all LSTM timestep outputs. This can better capture clinically
    important patterns at any point in the visit sequence.

    This variant is provided for ablation studies comparing
    final-hidden-state vs attention-pooled representations.

    Args:
        All arguments from ``ClinicalEncoder`` plus:
        attention_dim: Hidden dimension for the attention network
            (default: 128).

    Example:
        >>> encoder = ClinicalEncoderWithAttention()
        >>> x = torch.randn(4, 6, 5)
        >>> lengths = torch.tensor([6, 4, 3, 2])
        >>> emb = encoder(x, lengths)  # [4, 256]
    """

    def __init__(
        self,
        input_features: int = 5,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        embedding_dim: int = 256,
        attention_dim: int = 128,
    ) -> None:
        super().__init__(
            input_features=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            embedding_dim=embedding_dim,
        )

        self.temporal_attention = TemporalAttentionPooling(
            input_dim=self.lstm_output_dim,
            attention_dim=attention_dim,
        )

        logger.info(
            "ClinicalEncoderWithAttention initialised "
            "(attention_dim=%d)",
            attention_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode clinical sequence with attention pooling.

        Args:
            x: Clinical feature tensor ``[B, T, input_features]``.
            lengths: Actual sequence lengths ``[B]``.

        Returns:
            Embedding tensor ``[B, embedding_dim]``.
        """
        batch_size, max_seq_len, n_features = x.shape

        if n_features != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} input features, "
                f"got {n_features}"
            )

        lengths = lengths.clamp(min=1, max=max_seq_len)

        # Input projection: [B, T, 5] → [B, T, hidden_size]
        x = self.input_projection(x)

        # Sort by length for packing
        sorted_lengths, sort_indices = lengths.sort(descending=True)
        sorted_x = x[sort_indices]

        packed = pack_padded_sequence(
            sorted_x,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        # Bi-LSTM forward
        packed_output, (_h_n, _c_n) = self.lstm(packed)

        # Unpack to get all timestep outputs
        # lstm_outputs: [B, T, lstm_output_dim]
        lstm_outputs, _ = pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Unsort to restore original batch order
        _, unsort_indices = sort_indices.sort()
        lstm_outputs = lstm_outputs[unsort_indices]
        lengths_unsorted = lengths

        # Create validity mask: [B, T]
        max_len = lstm_outputs.size(1)
        time_range = torch.arange(
            max_len, device=x.device
        ).unsqueeze(0)
        mask = time_range < lengths_unsorted.unsqueeze(1)

        # Attention pooling: [B, T, D] → [B, D]
        pooled = self.temporal_attention(lstm_outputs, mask)

        # Dropout + projection: [B, D] → [B, embedding_dim]
        pooled = self.output_dropout(pooled)
        embedding = self.projection(pooled)

        return embedding

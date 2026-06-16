"""NeuroSense Unit Tests — Model Architecture.

Tests for all neural network modules:
- MRIEncoder: 3D ResNet-50 backbone with projection head
- ClinicalEncoder: Bi-LSTM for longitudinal clinical sequences
- CrossModalFusion: Cross-attention fusion module
- NeuroSenseModel: Full composite model
- ClassificationHead / ProgressionHead: Task heads
"""

from __future__ import annotations

import pytest
import torch

# ─── Test Configuration ───
BATCH_SIZE = 2
EMBED_DIM = 256
NUM_CLASSES = 3
MRI_SHAPE = (BATCH_SIZE, 1, 96, 96, 96)
CLINICAL_SHAPE = (BATCH_SIZE, 3, 5)  # 3 visits, 5 features


# ═════════════════════════════════════════════════════════════════
#  MRI Encoder Tests
# ═════════════════════════════════════════════════════════════════


class TestMRIEncoder:
    """Test suite for the 3D ResNet-50 MRI encoder."""

    def test_output_shape(self):
        """MRIEncoder produces [B, 256] embeddings from [B, 1, 96, 96, 96]."""
        from neurosense.models.mri_encoder import MRIEncoder

        encoder = MRIEncoder(embedding_dim=EMBED_DIM)
        mri = torch.randn(BATCH_SIZE, 1, 96, 96, 96)

        with torch.no_grad():
            emb = encoder(mri)

        assert emb.shape == (BATCH_SIZE, EMBED_DIM), (
            f"Expected ({BATCH_SIZE}, {EMBED_DIM}), got {emb.shape}"
        )

    def test_custom_embedding_dim(self):
        """MRIEncoder supports custom embedding dimensions."""
        from neurosense.models.mri_encoder import MRIEncoder

        custom_dim = 128
        encoder = MRIEncoder(embedding_dim=custom_dim)
        mri = torch.randn(1, 1, 96, 96, 96)

        with torch.no_grad():
            emb = encoder(mri)

        assert emb.shape == (1, custom_dim)

    def test_feature_dim_getter(self):
        """get_feature_dim() returns the correct embedding dimension."""
        from neurosense.models.mri_encoder import MRIEncoder

        encoder = MRIEncoder(embedding_dim=EMBED_DIM)
        assert encoder.get_feature_dim() == EMBED_DIM

    def test_gradient_flow(self):
        """Gradients flow through MRIEncoder during backward pass."""
        from neurosense.models.mri_encoder import MRIEncoder

        encoder = MRIEncoder(embedding_dim=EMBED_DIM)
        mri = torch.randn(1, 1, 96, 96, 96, requires_grad=True)

        emb = encoder(mri)
        loss = emb.sum()
        loss.backward()

        assert mri.grad is not None
        assert mri.grad.shape == mri.shape


# ═════════════════════════════════════════════════════════════════
#  Clinical Encoder Tests
# ═════════════════════════════════════════════════════════════════


class TestClinicalEncoder:
    """Test suite for the Bi-LSTM clinical encoder."""

    def test_output_shape(self):
        """ClinicalEncoder produces [B, 256] from [B, T, 5]."""
        from neurosense.models.clinical_encoder import ClinicalEncoder

        encoder = ClinicalEncoder(
            input_features=5,
            embedding_dim=EMBED_DIM,
            hidden_size=256,
            num_layers=2,
        )
        clinical = torch.randn(*CLINICAL_SHAPE)
        lengths = torch.tensor([3, 2])

        with torch.no_grad():
            emb = encoder(clinical, lengths)

        assert emb.shape == (BATCH_SIZE, EMBED_DIM)

    def test_single_visit(self):
        """ClinicalEncoder handles single-visit input [B, 1, 5]."""
        from neurosense.models.clinical_encoder import ClinicalEncoder

        encoder = ClinicalEncoder(
            input_features=5,
            embedding_dim=EMBED_DIM,
        )
        clinical = torch.randn(BATCH_SIZE, 1, 5)
        lengths = torch.ones(BATCH_SIZE, dtype=torch.long)

        with torch.no_grad():
            emb = encoder(clinical, lengths)

        assert emb.shape == (BATCH_SIZE, EMBED_DIM)

    def test_gradient_flow(self):
        """Gradients flow through ClinicalEncoder."""
        from neurosense.models.clinical_encoder import ClinicalEncoder

        encoder = ClinicalEncoder(
            input_features=5,
            embedding_dim=EMBED_DIM,
        )
        clinical = torch.randn(1, 2, 5, requires_grad=True)
        lengths = torch.tensor([2])

        emb = encoder(clinical, lengths)
        loss = emb.sum()
        loss.backward()

        assert clinical.grad is not None


# ═════════════════════════════════════════════════════════════════
#  Fusion Tests
# ═════════════════════════════════════════════════════════════════


class TestCrossModalFusion:
    """Test suite for cross-modal attention fusion."""

    def test_output_shape(self):
        """Fusion produces [B, 256] from two [B, 256] inputs."""
        from neurosense.models.fusion import CrossModalFusion

        fusion = CrossModalFusion(
            embed_dim=EMBED_DIM, num_heads=8
        )
        img_emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        clin_emb = torch.randn(BATCH_SIZE, EMBED_DIM)

        with torch.no_grad():
            fused = fusion(img_emb, clin_emb)

        assert fused.shape == (BATCH_SIZE, EMBED_DIM)

    def test_3d_input(self):
        """Fusion handles [B, 1, D] inputs (unsqueezed embeddings)."""
        from neurosense.models.fusion import CrossModalFusion

        fusion = CrossModalFusion(embed_dim=EMBED_DIM, num_heads=8)
        img_emb = torch.randn(BATCH_SIZE, 1, EMBED_DIM)
        clin_emb = torch.randn(BATCH_SIZE, 1, EMBED_DIM)

        with torch.no_grad():
            fused = fusion(img_emb, clin_emb)

        assert fused.shape == (BATCH_SIZE, EMBED_DIM)

    def test_attention_weights(self):
        """get_attention_weights() returns valid attention matrix."""
        from neurosense.models.fusion import CrossModalFusion

        num_heads = 8
        fusion = CrossModalFusion(
            embed_dim=EMBED_DIM, num_heads=num_heads
        )
        img_emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        clin_emb = torch.randn(BATCH_SIZE, EMBED_DIM)

        with torch.no_grad():
            weights = fusion.get_attention_weights(img_emb, clin_emb)

        assert weights is not None
        assert weights.shape == (BATCH_SIZE, num_heads, 1, 1)

    def test_concatenation_fusion(self):
        """ConcatenationFusion baseline produces correct shape."""
        from neurosense.models.fusion import ConcatenationFusion

        fusion = ConcatenationFusion(embed_dim=EMBED_DIM)
        img_emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        clin_emb = torch.randn(BATCH_SIZE, EMBED_DIM)

        with torch.no_grad():
            fused = fusion(img_emb, clin_emb)

        assert fused.shape == (BATCH_SIZE, EMBED_DIM)


# ═════════════════════════════════════════════════════════════════
#  Task Head Tests
# ═════════════════════════════════════════════════════════════════


class TestClassificationHead:
    """Test suite for the HD staging classification head."""

    def test_output_shape(self):
        """ClassificationHead produces [B, 3] logits."""
        from neurosense.models.mri_encoder import ClassificationHead

        head = ClassificationHead(
            input_dim=EMBED_DIM, num_classes=NUM_CLASSES
        )
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)

        with torch.no_grad():
            logits = head(emb)

        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


class TestProgressionHead:
    """Test suite for the UHDRS progression regression head."""

    def test_output_shape(self):
        """ProgressionHead produces [B, 2] deltas."""
        from neurosense.models.classifier import ProgressionHead

        head = ProgressionHead(input_dim=EMBED_DIM, output_dim=2)
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)

        with torch.no_grad():
            deltas = head(emb)

        assert deltas.shape == (BATCH_SIZE, 2)

    def test_risk_prediction(self):
        """predict_risk() returns valid risk categories."""
        from neurosense.models.classifier import ProgressionHead

        head = ProgressionHead(input_dim=EMBED_DIM)

        # Known delta values → expected risk categories
        deltas = torch.tensor([
            [1.0, 2.0],   # low (Δ < 3.0)
            [5.0, 7.0],   # medium (3.0 ≤ Δ < 8.0)
            [10.0, 15.0], # high (Δ ≥ 8.0)
        ])

        risks = head.predict_risk(deltas)

        assert len(risks) == 3
        assert risks[0] == "low"
        assert risks[1] == "medium"
        assert risks[2] == "high"


# ═════════════════════════════════════════════════════════════════
#  Full Model Tests
# ═════════════════════════════════════════════════════════════════


class TestNeuroSenseModel:
    """Test suite for the full NeuroSenseModel composite."""

    def test_full_multimodal_forward(self):
        """Full model produces all expected outputs."""
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
        )
        mri = torch.randn(*MRI_SHAPE)
        clinical = torch.randn(*CLINICAL_SHAPE)
        lengths = torch.tensor([3, 2])

        with torch.no_grad():
            outputs = model(mri, clinical, lengths)

        assert "logits" in outputs
        assert "probs" in outputs
        assert "stage_pred" in outputs
        assert "deltas" in outputs
        assert "risk" in outputs
        assert "embeddings" in outputs

        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)
        assert outputs["probs"].shape == (BATCH_SIZE, NUM_CLASSES)
        assert outputs["stage_pred"].shape == (BATCH_SIZE,)
        assert outputs["deltas"].shape == (BATCH_SIZE, 2)
        assert len(outputs["risk"]) == BATCH_SIZE

    def test_mri_only_mode(self):
        """Model works in MRI-only mode."""
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(
            use_mri=True, use_clinical=False
        )
        mri = torch.randn(*MRI_SHAPE)

        with torch.no_grad():
            outputs = model(mri=mri)

        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_clinical_only_mode(self):
        """Model works in clinical-only mode."""
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(
            use_mri=False, use_clinical=True
        )
        clinical = torch.randn(*CLINICAL_SHAPE)
        lengths = torch.tensor([3, 2])

        with torch.no_grad():
            outputs = model(clinical=clinical, clinical_lengths=lengths)

        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_no_modality_raises(self):
        """Model raises ValueError when both modalities disabled."""
        from neurosense.models.classifier import NeuroSenseModel

        with pytest.raises(ValueError, match="At least one modality"):
            NeuroSenseModel(use_mri=False, use_clinical=False)

    def test_freeze_unfreeze_encoders(self):
        """freeze_encoders/unfreeze_encoders toggle requires_grad."""
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel()

        model.freeze_encoders()
        for p in model.mri_encoder.parameters():
            assert not p.requires_grad
        for p in model.clinical_encoder.parameters():
            assert not p.requires_grad

        model.unfreeze_encoders()
        for p in model.mri_encoder.parameters():
            assert p.requires_grad
        for p in model.clinical_encoder.parameters():
            assert p.requires_grad

    def test_probabilities_sum_to_one(self):
        """Classification probabilities sum to ~1.0."""
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=False, use_clinical=True)
        clinical = torch.randn(1, 1, 5)
        lengths = torch.ones(1, dtype=torch.long)

        with torch.no_grad():
            outputs = model(clinical=clinical, clinical_lengths=lengths)

        prob_sum = outputs["probs"][0].sum().item()
        assert abs(prob_sum - 1.0) < 1e-5, (
            f"Probabilities sum to {prob_sum}, expected 1.0"
        )

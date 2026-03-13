"""Tests for the pretrained model splitter (PretrainedEncoder / PretrainedDecoder)."""

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.pretrained import (
    PretrainedDecoder,
    PretrainedEncoder,
    _extract_layers,
    _flatten_model_layers,
    _load_backbone,
)
from dendritic_modeling.networks.base import BaseNetwork


# ---------------------------------------------------------------------------
# Helper: build a minimal AlexNet-like model without downloading weights
# ---------------------------------------------------------------------------
class _MiniAlexNet(nn.Module):
    """Tiny model with the same named structure as torchvision AlexNet."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),  # 0
            nn.ReLU(),  # 1
            nn.MaxPool2d(2),  # 2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 0
            nn.Linear(32, 16),  # 1  (fc6)
            nn.ReLU(),  # 2
            nn.Dropout(),  # 3
            nn.Linear(16, 16),  # 4  (fc7)
            nn.ReLU(),  # 5
            nn.Linear(16, 10),  # 6  (fc8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Tests for _flatten_model_layers
# ---------------------------------------------------------------------------
class TestFlattenModelLayers:
    def test_expands_sequential(self):
        model = _MiniAlexNet()
        layers = _flatten_model_layers(model)
        names = [n for n, _ in layers]

        # Sequential children should be expanded
        assert "features.0" in names
        assert "features.1" in names
        assert "features.2" in names
        assert "avgpool" in names
        assert "classifier.0" in names
        assert "classifier.6" in names

        # Top-level Sequential names should NOT appear
        assert "features" not in names
        assert "classifier" not in names

    def test_preserves_order(self):
        model = _MiniAlexNet()
        layers = _flatten_model_layers(model)
        names = [n for n, _ in layers]

        # features before avgpool before classifier
        assert names.index("features.0") < names.index("avgpool")
        assert names.index("avgpool") < names.index("classifier.0")


# ---------------------------------------------------------------------------
# Tests for _extract_layers
# ---------------------------------------------------------------------------
class TestExtractLayers:
    def test_split_after(self):
        model = _MiniAlexNet()
        modules, names = _extract_layers(model, split_after="classifier.2")

        assert "classifier.2" in names
        assert "classifier.4" not in names
        assert "classifier.6" not in names

    def test_split_from(self):
        model = _MiniAlexNet()
        modules, names = _extract_layers(model, split_from="classifier.6")

        assert names == ["classifier.6"]
        assert isinstance(modules[0], nn.Linear)

    def test_drop_layers(self):
        model = _MiniAlexNet()
        modules, names = _extract_layers(
            model,
            split_after="classifier.2",
            drop_layers=["classifier.0"],
        )

        assert "classifier.0" not in names
        # fc6 and ReLU should still be there
        assert "classifier.1" in names
        assert "classifier.2" in names

    def test_invalid_split_after_raises(self):
        model = _MiniAlexNet()
        with pytest.raises(ValueError, match="not found"):
            _extract_layers(model, split_after="nonexistent_layer")

    def test_invalid_split_from_raises(self):
        model = _MiniAlexNet()
        with pytest.raises(ValueError, match="not found"):
            _extract_layers(model, split_from="nonexistent_layer")

    def test_neither_split_raises(self):
        model = _MiniAlexNet()
        with pytest.raises(ValueError, match="Either split_after or split_from"):
            _extract_layers(model)

    def test_split_after_avgpool(self):
        model = _MiniAlexNet()
        modules, names = _extract_layers(model, split_after="avgpool")

        assert names[-1] == "avgpool"
        assert "classifier.0" not in names


# ---------------------------------------------------------------------------
# Tests for PretrainedEncoder (using real torchvision AlexNet)
# ---------------------------------------------------------------------------
class TestPretrainedEncoderReal:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        """Skip if torchvision models can't be loaded."""
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_alexnet_k1_encoder(self):
        """K=1: features + fc6 -> 4096-dim output."""
        enc = PretrainedEncoder(
            backbone="alexnet",
            weights=None,  # random init to avoid download
            split_after="classifier.2",
            freeze=True,
            drop_layers=["classifier.0", "classifier.3"],
        )

        assert enc.output_dim == 4096
        assert enc.frozen is True

        # All params frozen
        for p in enc.parameters():
            assert not p.requires_grad

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        out = enc(x)
        assert out.shape == (2, 4096)

    def test_alexnet_k2_encoder(self):
        """K=2: features only -> 9216-dim output."""
        enc = PretrainedEncoder(
            backbone="alexnet",
            weights=None,
            split_after="avgpool",
            freeze=True,
        )

        assert enc.output_dim == 9216

        x = torch.randn(1, 3, 224, 224)
        out = enc(x)
        assert out.shape == (1, 9216)

    def test_unfrozen_encoder(self):
        """Encoder with freeze=False has trainable params."""
        enc = PretrainedEncoder(
            backbone="alexnet",
            weights=None,
            split_after="classifier.2",
            freeze=False,
        )

        assert enc.frozen is False
        trainable = [p for p in enc.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_resnet18_encoder(self):
        """ResNet18: split after avgpool -> 512-dim."""
        try:
            from torchvision.models import get_model

            get_model("resnet18", weights=None)
        except Exception:
            pytest.skip("resnet18 not available")

        enc = PretrainedEncoder(
            backbone="resnet18",
            weights=None,
            split_after="avgpool",
            freeze=True,
        )

        assert enc.output_dim == 512

        x = torch.randn(1, 3, 224, 224)
        out = enc(x)
        assert out.shape == (1, 512)

    def test_3d_input_adds_batch_dim(self):
        """Single image (C, H, W) should work."""
        enc = PretrainedEncoder(
            backbone="alexnet",
            weights=None,
            split_after="classifier.2",
            freeze=True,
            drop_layers=["classifier.0"],
        )

        x = torch.randn(3, 224, 224)
        out = enc(x)
        assert out.shape == (4096,)


# ---------------------------------------------------------------------------
# Tests for PretrainedDecoder (using real torchvision AlexNet)
# ---------------------------------------------------------------------------
class TestPretrainedDecoderReal:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_alexnet_fc8_decoder(self):
        """fc8: Linear(4096 -> 1000)."""
        dec = PretrainedDecoder(
            backbone="alexnet",
            weights=None,
            split_from="classifier.6",
            freeze=False,
        )

        assert dec.output_dim == 1000

        x = torch.randn(2, 4096)
        out = dec(x)
        assert out.shape == (2, 1000)

    def test_frozen_decoder(self):
        dec = PretrainedDecoder(
            backbone="alexnet",
            weights=None,
            split_from="classifier.6",
            freeze=True,
        )

        for p in dec.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Tests for factory registration
# ---------------------------------------------------------------------------
class TestFactoryRegistration:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_factory_pretrained_encoder(self):
        from dendritic_modeling.networks.architectures.factory import get_architecture

        enc = get_architecture(
            "pretrained",
            {
                "backbone": "alexnet",
                "weights": None,
                "split_after": "classifier.2",
                "freeze": True,
                "drop_layers": ["classifier.0", "classifier.3"],
            },
        )

        assert isinstance(enc, PretrainedEncoder)
        assert enc.output_dim == 4096

    def test_factory_pretrained_decoder(self):
        from dendritic_modeling.networks.architectures.factory import get_architecture

        dec = get_architecture(
            "pretrained_decoder",
            {
                "backbone": "alexnet",
                "weights": None,
                "split_from": "classifier.6",
                "freeze": False,
            },
        )

        assert isinstance(dec, PretrainedDecoder)
        assert dec.output_dim == 1000


# ---------------------------------------------------------------------------
# Tests for _load_backbone
# ---------------------------------------------------------------------------
class TestLoadBackbone:
    def test_none_weights(self):
        """weights=None creates model with random init."""
        model = _load_backbone("alexnet", weights=None)
        assert model is not None
        # Should have the standard AlexNet structure
        assert hasattr(model, "features")
        assert hasattr(model, "classifier")

    def test_local_file_path(self, tmp_path):
        """Loading from a local .pt file."""
        model = _load_backbone("alexnet", weights=None)
        path = str(tmp_path / "model.pt")
        torch.save(model.state_dict(), path)

        loaded = _load_backbone("alexnet", weights=path)
        assert loaded is not None

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# Tests for decoder input validation (ValueError on mismatch)
# ---------------------------------------------------------------------------
class TestDecoderInputValidation:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_decoder_mismatch_raises(self):
        """input_dim that doesn't match first Linear.in_features should raise."""
        with pytest.raises(ValueError, match="does not match"):
            PretrainedDecoder(
                backbone="alexnet",
                weights=None,
                split_from="classifier.6",
                freeze=False,
                input_dim=999,  # AlexNet fc8 expects 4096
            )

    def test_decoder_correct_input_dim_ok(self):
        """Correct input_dim should not raise."""
        dec = PretrainedDecoder(
            backbone="alexnet",
            weights=None,
            split_from="classifier.6",
            freeze=False,
            input_dim=4096,
        )
        assert dec.output_dim == 1000

    def test_decoder_none_input_dim_ok(self):
        """input_dim=None skips validation."""
        dec = PretrainedDecoder(
            backbone="alexnet",
            weights=None,
            split_from="classifier.6",
            freeze=False,
            input_dim=None,
        )
        assert dec.output_dim == 1000


# ---------------------------------------------------------------------------
# End-to-end smoke test: encoder -> core -> decoder forward pass
# ---------------------------------------------------------------------------
class _SimpleCore(BaseNetwork):
    """Minimal core network stand-in for testing the pipeline."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)

    def get_output_dim(self):
        return self.output_dim


class TestEndToEndPipeline:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_alexnet_encoder_core_decoder_forward(self):
        """Full encoder -> core -> decoder pipeline on a random batch."""
        encoder = PretrainedEncoder(
            backbone="alexnet",
            weights=None,
            split_after="classifier.2",
            freeze=True,
            drop_layers=["classifier.0", "classifier.3"],
        )
        core = _SimpleCore(encoder.output_dim, 256)
        decoder = PretrainedDecoder(
            backbone="alexnet",
            weights=None,
            split_from="classifier.6",
            freeze=False,
            input_dim=None,  # fc8 expects 4096, but we use core output=256
        )
        # Replace decoder's first linear to accept core output
        decoder.layers = nn.Sequential(nn.Linear(256, 1000))
        decoder.output_dim = 1000

        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            encoded = encoder(x)
        assert encoded.shape == (4, 4096)

        core_out = core(encoded)
        assert core_out.shape == (4, 256)

        logits = decoder(core_out)
        assert logits.shape == (4, 1000)

    def test_alexnet_k2_encoder_core_decoder_forward(self):
        """K=2 split: features only -> core -> decoder."""
        encoder = PretrainedEncoder(
            backbone="alexnet",
            weights=None,
            split_after="avgpool",
            freeze=True,
        )
        assert encoder.output_dim == 9216

        core = _SimpleCore(9216, 128)
        decoder_layer = nn.Linear(128, 10)

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            encoded = encoder(x)
        core_out = core(encoded)
        logits = decoder_layer(core_out)
        assert logits.shape == (2, 10)

    def test_frozen_encoder_no_grad(self):
        """Frozen encoder should not accumulate gradients during pipeline."""
        encoder = PretrainedEncoder(
            backbone="alexnet",
            weights=None,
            split_after="classifier.2",
            freeze=True,
            drop_layers=["classifier.0", "classifier.3"],
        )
        core = _SimpleCore(encoder.output_dim, 100)

        x = torch.randn(2, 3, 224, 224)
        encoded = encoder(x)
        out = core(encoded)
        loss = out.sum()
        loss.backward()

        # Encoder params should have no gradient
        for p in encoder.parameters():
            assert p.grad is None
        # Core params should have gradient
        for p in core.parameters():
            assert p.grad is not None

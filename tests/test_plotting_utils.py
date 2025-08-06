"""
Tests for plotting utilities to improve coverage.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

# Import plotting utilities
from dendritic_modeling.plotting.core.utils import (
    dendritic_activations_todict,
    einet_activations_todict,
    visualize_graph,
)


class TestPlottingUtils:
    """Test plotting utility functions."""

    def test_visualize_graph_networkx_available(self):
        """Test visualize_graph when NetworkX is available."""
        with patch("dendritic_modeling.plotting.core.utils.NETWORKX_AVAILABLE", True):
            with patch("dendritic_modeling.plotting.core.utils.nx") as mock_nx:
                with patch("matplotlib.pyplot.figure") as mock_figure:
                    mock_graph = MagicMock()
                    mock_graph.nodes.return_value = [
                        (1, {"node_type": "layer"}),
                        (2, {"node_type": "branch"}),
                    ]
                    mock_nx.spring_layout.return_value = {1: (0, 0), 2: (1, 1)}

                    # Should not raise exception
                    visualize_graph(mock_graph, mode="info")

                    # The function calls figure multiple times, so just check it was called
                    assert mock_figure.call_count >= 1
                    mock_nx.spring_layout.assert_called_once()

    def test_visualize_graph_networkx_unavailable(self):
        """Test visualize_graph when NetworkX is unavailable."""
        with patch("dendritic_modeling.plotting.core.utils.NETWORKX_AVAILABLE", False):
            with patch(
                "dendritic_modeling.plotting.core.utils.logging"
            ) as mock_logging:
                mock_graph = MagicMock()

                # Should log warning and return
                result = visualize_graph(mock_graph, mode="info")

                assert result is None
                mock_logging.warning.assert_called_once()

    def test_visualize_graph_none_graph(self):
        """Test visualize_graph with None graph."""
        with patch("dendritic_modeling.plotting.core.utils.NETWORKX_AVAILABLE", True):
            with patch(
                "dendritic_modeling.plotting.core.utils.logging"
            ) as mock_logging:

                # Should log warning and return
                result = visualize_graph(None, mode="info")

                assert result is None
                mock_logging.warning.assert_called_once()

    def test_visualize_graph_different_modes(self):
        """Test visualize_graph with different modes."""
        with patch("dendritic_modeling.plotting.core.utils.NETWORKX_AVAILABLE", True):
            with patch("dendritic_modeling.plotting.core.utils.nx") as mock_nx:
                with patch("matplotlib.pyplot.figure") as mock_figure:
                    mock_graph = MagicMock()
                    mock_graph.nodes.return_value = [(1, {"node_type": "layer"})]
                    mock_nx.spring_layout.return_value = {1: (0, 0)}

                    # Test different modes
                    for mode in ["info", "simple", "detailed"]:
                        visualize_graph(mock_graph, mode=mode)

                    # Function calls figure multiple times per call, so just check it was called
                    assert mock_figure.call_count >= 3

    def test_dendritic_activations_todict_basic(self):
        """Test dendritic_activations_todict with basic inputs."""
        # Create mock dendrinet
        mock_dendrinet = MagicMock()
        mock_dendrinet.branches = [MagicMock() for _ in range(3)]

        # Create mock inputs
        exc_input_list = [torch.randn(10, 5) for _ in range(2)]
        inh_input_list = [torch.randn(10, 3) for _ in range(2)]

        with patch.object(mock_dendrinet, "forward", return_value=torch.randn(10, 2)):
            result = dendritic_activations_todict(
                mock_dendrinet, exc_input_list, inh_input_list
            )

            assert isinstance(result, dict)

    def test_dendritic_activations_todict_empty_inputs(self):
        """Test dendritic_activations_todict with empty inputs."""
        mock_dendrinet = MagicMock()
        mock_dendrinet.branches = []

        result = dendritic_activations_todict(mock_dendrinet, [], [])

        assert isinstance(result, dict)

    def test_dendritic_activations_todict_mismatched_inputs(self):
        """Test dendritic_activations_todict with mismatched input lengths."""
        mock_dendrinet = MagicMock()
        mock_dendrinet.branches = [MagicMock()]

        exc_input_list = [torch.randn(10, 5)]
        inh_input_list = [torch.randn(10, 3), torch.randn(10, 3)]  # Different length

        result = dendritic_activations_todict(
            mock_dendrinet, exc_input_list, inh_input_list
        )

        assert isinstance(result, dict)

    def test_einet_activations_todict_basic(self):
        """Test einet_activations_todict with basic inputs."""
        # Create mock einet
        mock_einet = MagicMock()

        # Create mock input list
        input_list = [torch.randn(10, 20) for _ in range(3)]

        with patch.object(mock_einet, "forward", return_value=torch.randn(10, 5)):
            result = einet_activations_todict(mock_einet, input_list)

            assert isinstance(result, dict)

    def test_einet_activations_todict_empty_inputs(self):
        """Test einet_activations_todict with empty inputs."""
        mock_einet = MagicMock()

        result = einet_activations_todict(mock_einet, [])

        assert isinstance(result, dict)

    def test_einet_activations_todict_single_input(self):
        """Test einet_activations_todict with single input."""
        mock_einet = MagicMock()
        input_list = [torch.randn(5, 10)]

        with patch.object(mock_einet, "forward", return_value=torch.randn(5, 3)):
            result = einet_activations_todict(mock_einet, input_list)

            assert isinstance(result, dict)

    def test_einet_activations_todict_large_inputs(self):
        """Test einet_activations_todict with large input list."""
        mock_einet = MagicMock()
        input_list = [torch.randn(2, 8) for _ in range(10)]

        with patch.object(mock_einet, "forward", return_value=torch.randn(2, 4)):
            result = einet_activations_todict(mock_einet, input_list)

            assert isinstance(result, dict)


class TestPlottingUtilsEdgeCases:
    """Test edge cases for plotting utilities."""

    def test_visualize_graph_complex_structure(self):
        """Test visualize_graph with complex graph structure."""
        with patch("dendritic_modeling.plotting.core.utils.NETWORKX_AVAILABLE", True):
            with patch("dendritic_modeling.plotting.core.utils.nx") as mock_nx:
                with patch("matplotlib.pyplot.figure"):
                    mock_graph = MagicMock()
                    # Complex node structure
                    mock_graph.nodes.return_value = [
                        (1, {"node_type": "layer", "label": "input"}),
                        (2, {"node_type": "branch", "label": "dendrite1"}),
                        (3, {"node_type": "layer", "label": "output"}),
                        (4, {"node_type": "unknown"}),  # Unknown type
                        (5, {}),  # No node_type
                    ]
                    mock_nx.spring_layout.return_value = {
                        i: (i, i) for i in range(1, 6)
                    }

                    visualize_graph(mock_graph, mode="complex")

                    mock_nx.spring_layout.assert_called_once()

    def test_dendritic_activations_with_exceptions(self):
        """Test dendritic_activations_todict with exceptions during execution."""
        mock_dendrinet = MagicMock()
        mock_dendrinet.layer_sizes = [10, 5]
        mock_dendrinet.n_soma = 2
        mock_dendrinet.n_branch_layers = 1
        mock_dendrinet.branch_layers = [MagicMock()]
        mock_dendrinet.branch_layers[0].side_effect = RuntimeError(
            "Branch layer failed"
        )

        exc_input_list = [torch.randn(5, 10)]
        inh_input_list = [torch.randn(5, 8)]

        # Should propagate the RuntimeError from branch_layers
        with pytest.raises(RuntimeError):
            dendritic_activations_todict(mock_dendrinet, exc_input_list, inh_input_list)

    def test_einet_activations_with_none_inputs(self):
        """Test einet_activations_todict with None in input list."""
        mock_einet = MagicMock()
        input_list = [torch.randn(3, 5), None, torch.randn(3, 5)]

        # Function doesn't handle None values gracefully - will raise AttributeError
        with pytest.raises(AttributeError):
            einet_activations_todict(mock_einet, input_list)

    def test_dendritic_activations_different_tensor_shapes(self):
        """Test dendritic_activations_todict with different tensor shapes."""
        mock_dendrinet = MagicMock()
        mock_dendrinet.branches = [MagicMock() for _ in range(2)]

        # Different shapes
        exc_input_list = [
            torch.randn(5, 10),
            torch.randn(3, 15),  # Different batch size and features
        ]
        inh_input_list = [
            torch.randn(5, 8),
            torch.randn(3, 12),  # Different batch size and features
        ]

        with patch.object(mock_dendrinet, "forward", return_value=torch.randn(5, 4)):
            result = dendritic_activations_todict(
                mock_dendrinet, exc_input_list, inh_input_list
            )

            assert isinstance(result, dict)

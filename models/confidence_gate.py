"""
Sample-Adaptive Confidence Gate for HVNet.

This module implements the confidence-gating stage described in the
HVNet manuscript.

The gate input is:

    g = [h_clin; h_vis; u_c; u_v]

where:

    h_clin : clinical representation
    h_vis  : visual representation
    u_c    : clinical uncertainty
    u_v    : visual uncertainty

For the manuscript configuration:

    dim(h_clin) = 256
    dim(h_vis)  = 256

Therefore:

    dim(g) = 256 + 256 + 1 + 1 = 514

The gating network produces three sample-specific fusion weights:

    [w_clin, w_vis, w_ds]
        = softmax(MLP_gate(g))

with:

    w_clin >= 0
    w_vis  >= 0
    w_ds   >= 0

and:

    w_clin + w_vis + w_ds = 1

The final HVNet predictive distribution is:

    p_final
        = w_clin * p_clin
        + w_vis  * p_vis
        + w_ds   * p_ds

The output-layer bias is initialized as:

    [1.0, -0.5, -0.5]

to provide a mild initial preference toward the clinical hypothesis
branch, as specified in the manuscript implementation.
"""

from typing import Dict

import torch
import torch.nn as nn


class SampleAdaptiveConfidenceGate(nn.Module):
    """
    Sample-adaptive confidence gate used in the Arbitrate stage of HVNet.

    Parameters
    ----------
    clinical_dim : int
        Dimension of the clinical representation h_clin.
        Manuscript configuration: 256.

    visual_dim : int
        Dimension of the visual representation h_vis.
        Manuscript configuration: 256.

    gate_hidden_dim : int
        Hidden dimension d_gate of the gating MLP.

        This corresponds to:

            514 -> d_gate -> 3

        The exact value should match the original training implementation.

    dropout : float
        Dropout probability inside the gating MLP.

    activation : str
        Non-linear activation used in the hidden layer.

        Supported:
            "gelu"
            "relu"
    """

    NUM_GATE_BRANCHES = 3

    def __init__(
        self,
        clinical_dim: int = 256,
        visual_dim: int = 256,
        gate_hidden_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.clinical_dim = clinical_dim
        self.visual_dim = visual_dim
        self.gate_hidden_dim = gate_hidden_dim

        # --------------------------------------------------------
        # Gate input:
        #
        # g = [h_clin; h_vis; u_c; u_v]
        #
        # For the manuscript configuration:
        #
        # 256 + 256 + 1 + 1 = 514
        # --------------------------------------------------------

        self.gate_input_dim = (
            clinical_dim
            + visual_dim
            + 2
        )

        # --------------------------------------------------------
        # Activation
        # --------------------------------------------------------

        activation = activation.lower()

        if activation == "gelu":
            activation_layer = nn.GELU()

        elif activation == "relu":
            activation_layer = nn.ReLU(inplace=False)

        else:
            raise ValueError(
                "Unsupported activation. "
                "Expected 'gelu' or 'relu', "
                f"but received '{activation}'."
            )

        # --------------------------------------------------------
        # MLP_gate:
        #
        # 514 -> d_gate -> 3
        # --------------------------------------------------------

        self.hidden_layer = nn.Linear(
            in_features=self.gate_input_dim,
            out_features=gate_hidden_dim,
        )

        self.activation = activation_layer

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.output_layer = nn.Linear(
            in_features=gate_hidden_dim,
            out_features=self.NUM_GATE_BRANCHES,
        )

        # --------------------------------------------------------
        # Manuscript-specified output bias initialization:
        #
        # [clinical, visual, DS]
        #     [1.0, -0.5, -0.5]
        # --------------------------------------------------------

        self._initialize_gate()

    def _initialize_gate(self) -> None:
        """
        Initialize the final gating layer.

        The final bias follows the manuscript:

            [1.0, -0.5, -0.5]

        corresponding to:

            clinical branch
            visual branch
            DS-fusion branch
        """

        nn.init.xavier_uniform_(
            self.output_layer.weight
        )

        with torch.no_grad():

            self.output_layer.bias.copy_(
                torch.tensor(
                    [1.0, -0.5, -0.5],
                    dtype=self.output_layer.bias.dtype,
                    device=self.output_layer.bias.device,
                )
            )

    @staticmethod
    def _prepare_uncertainty(
        uncertainty: torch.Tensor,
        name: str,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Convert uncertainty to shape [B, 1].

        Accepted input shapes:

            [B]
            [B, 1]

        Parameters
        ----------
        uncertainty : torch.Tensor
            Modality-specific uncertainty.

        name : str
            Name used in error messages.

        batch_size : int
            Expected batch size.

        Returns
        -------
        torch.Tensor
            Uncertainty tensor with shape [B, 1].
        """

        if not torch.is_tensor(uncertainty):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if uncertainty.ndim == 1:
            uncertainty = uncertainty.unsqueeze(-1)

        if (
            uncertainty.ndim != 2
            or uncertainty.size(1) != 1
        ):
            raise ValueError(
                f"{name} must have shape [B] or [B, 1], "
                f"but received {tuple(uncertainty.shape)}."
            )

        if uncertainty.size(0) != batch_size:
            raise ValueError(
                f"{name} batch size does not match "
                "the feature representations."
            )

        return uncertainty

    @staticmethod
    def _validate_probability_distribution(
        probability: torch.Tensor,
        name: str,
        batch_size: int,
        num_classes: int,
    ) -> None:
        """
        Validate the shape of a predictive distribution.

        Expected shape:

            [B, K]
        """

        if not torch.is_tensor(probability):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if probability.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, K], "
                f"but received {tuple(probability.shape)}."
            )

        if probability.size(0) != batch_size:
            raise ValueError(
                f"{name} has an inconsistent batch size."
            )

        if probability.size(1) != num_classes:
            raise ValueError(
                f"{name} has {probability.size(1)} classes, "
                f"but expected {num_classes}."
            )

    def compute_gate_weights(
        self,
        h_clin: torch.Tensor,
        h_vis: torch.Tensor,
        u_clin: torch.Tensor,
        u_vis: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sample-specific gating weights.

        Implements:

            g = [h_clin; h_vis; u_c; u_v]

            [w_clin, w_vis, w_ds]
                = softmax(MLP_gate(g))

        Parameters
        ----------
        h_clin : torch.Tensor
            Clinical representation.
            Shape: [B, clinical_dim].

        h_vis : torch.Tensor
            Visual representation.
            Shape: [B, visual_dim].

        u_clin : torch.Tensor
            Clinical uncertainty u_c.
            Shape: [B] or [B, 1].

        u_vis : torch.Tensor
            Visual uncertainty u_v.
            Shape: [B] or [B, 1].

        Returns
        -------
        dict
            gate_input:
                Concatenated gating representation g.

            gate_logits:
                Raw three-dimensional gating logits.

            weights:
                Softmax-normalized weights
                [w_clin, w_vis, w_ds].

            w_clin:
                Clinical weight.

            w_vis:
                Visual weight.

            w_ds:
                DS-fusion weight.
        """

        # --------------------------------------------------------
        # Validate feature representations
        # --------------------------------------------------------

        if h_clin.ndim != 2:
            raise ValueError(
                "h_clin must have shape "
                "[B, clinical_dim]."
            )

        if h_vis.ndim != 2:
            raise ValueError(
                "h_vis must have shape "
                "[B, visual_dim]."
            )

        batch_size = h_clin.size(0)

        if h_vis.size(0) != batch_size:
            raise ValueError(
                "h_clin and h_vis must have "
                "the same batch size."
            )

        if h_clin.size(1) != self.clinical_dim:
            raise ValueError(
                f"Expected h_clin dimension "
                f"{self.clinical_dim}, "
                f"but received {h_clin.size(1)}."
            )

        if h_vis.size(1) != self.visual_dim:
            raise ValueError(
                f"Expected h_vis dimension "
                f"{self.visual_dim}, "
                f"but received {h_vis.size(1)}."
            )

        # --------------------------------------------------------
        # Prepare uncertainty scalars
        # --------------------------------------------------------

        u_clin = self._prepare_uncertainty(
            u_clin,
            name="u_clin",
            batch_size=batch_size,
        )

        u_vis = self._prepare_uncertainty(
            u_vis,
            name="u_vis",
            batch_size=batch_size,
        )

        # --------------------------------------------------------
        # g = [h_clin; h_vis; u_c; u_v]
        #
        # [B, 256 + 256 + 1 + 1]
        # = [B, 514]
        # --------------------------------------------------------

        gate_input = torch.cat(
            [
                h_clin,
                h_vis,
                u_clin,
                u_vis,
            ],
            dim=-1,
        )

        # --------------------------------------------------------
        # MLP_gate(g)
        #
        # 514 -> d_gate -> 3
        # --------------------------------------------------------

        hidden = self.hidden_layer(
            gate_input
        )

        hidden = self.activation(
            hidden
        )

        hidden = self.dropout(
            hidden
        )

        gate_logits = self.output_layer(
            hidden
        )

        # --------------------------------------------------------
        # [w_clin, w_vis, w_ds]
        #     = softmax(MLP_gate(g))
        #
        # Therefore all weights are non-negative and:
        #
        # w_clin + w_vis + w_ds = 1
        # --------------------------------------------------------

        weights = torch.softmax(
            gate_logits,
            dim=-1,
        )

        w_clin = weights[:, 0:1]
        w_vis = weights[:, 1:2]
        w_ds = weights[:, 2:3]

        return {
            "gate_input": gate_input,
            "gate_logits": gate_logits,
            "weights": weights,
            "w_clin": w_clin,
            "w_vis": w_vis,
            "w_ds": w_ds,
        }

    def forward(
        self,
        h_clin: torch.Tensor,
        h_vis: torch.Tensor,
        u_clin: torch.Tensor,
        u_vis: torch.Tensor,
        p_clin: torch.Tensor,
        p_vis: torch.Tensor,
        p_ds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform sample-adaptive confidence gating and final fusion.

        Implements:

            g = [h_clin; h_vis; u_c; u_v]

            [w_clin, w_vis, w_ds]
                = softmax(MLP_gate(g))

        followed by:

            p_final
                = w_clin * p_clin
                + w_vis * p_vis
                + w_ds * p_ds

        Parameters
        ----------
        h_clin : torch.Tensor
            Clinical representation.
            Shape: [B, clinical_dim].

        h_vis : torch.Tensor
            Visual representation.
            Shape: [B, visual_dim].

        u_clin : torch.Tensor
            Clinical uncertainty.
            Shape: [B] or [B, 1].

        u_vis : torch.Tensor
            Visual uncertainty.
            Shape: [B] or [B, 1].

        p_clin : torch.Tensor
            Clinical predictive distribution.
            Shape: [B, K].

        p_vis : torch.Tensor
            Visual predictive distribution.
            Shape: [B, K].

        p_ds : torch.Tensor
            Dempster--Shafer fused predictive distribution.
            Shape: [B, K].

        Returns
        -------
        dict
            p_final:
                Final HVNet predictive distribution.

            weights:
                Three confidence-gate weights.

            w_clin:
                Clinical branch weight.

            w_vis:
                Visual branch weight.

            w_ds:
                DS-fusion branch weight.

            gate_logits:
                Raw gate logits.

            gate_input:
                Concatenated gate representation.
        """

        # --------------------------------------------------------
        # Calculate confidence-gating weights
        # --------------------------------------------------------

        gate_outputs = self.compute_gate_weights(
            h_clin=h_clin,
            h_vis=h_vis,
            u_clin=u_clin,
            u_vis=u_vis,
        )

        batch_size = h_clin.size(0)

        # Infer the number of classes from p_clin.
        if p_clin.ndim != 2:
            raise ValueError(
                "p_clin must have shape [B, K]."
            )

        num_classes = p_clin.size(1)

        # --------------------------------------------------------
        # Validate predictive distributions
        # --------------------------------------------------------

        self._validate_probability_distribution(
            p_clin,
            name="p_clin",
            batch_size=batch_size,
            num_classes=num_classes,
        )

        self._validate_probability_distribution(
            p_vis,
            name="p_vis",
            batch_size=batch_size,
            num_classes=num_classes,
        )

        self._validate_probability_distribution(
            p_ds,
            name="p_ds",
            batch_size=batch_size,
            num_classes=num_classes,
        )

        w_clin = gate_outputs["w_clin"]
        w_vis = gate_outputs["w_vis"]
        w_ds = gate_outputs["w_ds"]

        # --------------------------------------------------------
        # Final HVNet prediction:
        #
        # p_final =
        #     w_clin * p_clin
        #     + w_vis * p_vis
        #     + w_ds * p_ds
        # --------------------------------------------------------

        p_final = (
            w_clin * p_clin
            + w_vis * p_vis
            + w_ds * p_ds
        )

        return {
            "p_final": p_final,
            "weights": gate_outputs["weights"],
            "w_clin": w_clin,
            "w_vis": w_vis,
            "w_ds": w_ds,
            "gate_logits": gate_outputs["gate_logits"],
            "gate_input": gate_outputs["gate_input"],
        }


# ============================================================
# Short aliases
# ============================================================

ConfidenceGate = SampleAdaptiveConfidenceGate
AdaptiveConfidenceGate = SampleAdaptiveConfidenceGate


# ============================================================
# Minimal sanity check
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 4
    num_classes = 3

    gate = SampleAdaptiveConfidenceGate(
        clinical_dim=256,
        visual_dim=256,
        gate_hidden_dim=128,
        dropout=0.0,
        activation="gelu",
    )

    # Example clinical and visual representations
    h_clin = torch.randn(
        batch_size,
        256,
    )

    h_vis = torch.randn(
        batch_size,
        256,
    )

    # Example modality uncertainties
    u_clin = torch.rand(
        batch_size,
        1,
    )

    u_vis = torch.rand(
        batch_size,
        1,
    )

    # Example predictive distributions
    p_clin = torch.softmax(
        torch.randn(batch_size, num_classes),
        dim=-1,
    )

    p_vis = torch.softmax(
        torch.randn(batch_size, num_classes),
        dim=-1,
    )

    p_ds = torch.softmax(
        torch.randn(batch_size, num_classes),
        dim=-1,
    )

    outputs = gate(
        h_clin=h_clin,
        h_vis=h_vis,
        u_clin=u_clin,
        u_vis=u_vis,
        p_clin=p_clin,
        p_vis=p_vis,
        p_ds=p_ds,
    )

    print(
        "Gate input shape:",
        outputs["gate_input"].shape,
    )

    print(
        "Gate weights shape:",
        outputs["weights"].shape,
    )

    print(
        "Final prediction shape:",
        outputs["p_final"].shape,
    )

    print(
        "\nGate weights:"
    )

    print(
        outputs["weights"]
    )

    print(
        "\nWeight sums:"
    )

    print(
        outputs["weights"].sum(dim=-1)
    )

    print(
        "\nFinal probability sums:"
    )

    print(
        outputs["p_final"].sum(dim=-1)
    )

"""
Dempster--Shafer Evidence Fusion for HVNet.

This module implements the uncertainty-aware evidential fusion stage
described in the HVNet manuscript.

Clinical and visual branches produce Dirichlet concentration parameters:

    alpha_c
    alpha_v

The evidence of each modality is first temperature calibrated:

    alpha'_m = (alpha_m - 1) * tau + 1

where:

    m in {c, v}

and tau is the evidence calibration temperature.

For each modality, the Dirichlet parameters are converted into
belief masses and uncertainty:

    S_m = sum_k alpha'_{m,k}

    b_{m,k} = (alpha'_{m,k} - 1) / S_m

    u_m = K / S_m

where K is the number of classes.

The clinical and visual evidence is then combined using
Dempster--Shafer evidence fusion:

    tilde_b_k =
        b_{c,k} * b_{v,k}
        + b_{c,k} * u_v
        + u_c * b_{v,k}

The cross-class conflict mass is:

    C = sum_{i != j} b_{c,i} * b_{v,j}

The fused belief and uncertainty are:

    b_ds,k = tilde_b_k / (1 - C)

    u_ds = (u_c * u_v) / (1 - C)

The final DS predictive distribution is obtained by normalizing
the fused belief mass:

    p_ds,k = b_ds,k / sum_j b_ds,j

The implementation below follows these equations directly.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DempsterShaferFusion(nn.Module):
    """
    Dempster--Shafer evidential fusion module used by HVNet.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
        HVNet uses:
            K = 3
        corresponding to:
            Normal, OAG, ACG.

    temperature : float
        Evidence calibration temperature tau.

        The calibrated Dirichlet parameters are:

            alpha' = (alpha - 1) * tau + 1

        Manuscript configuration:
            tau = 1.0

    eps : float
        Numerical stability constant used when dividing by quantities
        that may approach zero.
    """

    def __init__(
        self,
        num_classes: int = 3,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be greater than 0."
            )

        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = eps

    # ============================================================
    # Input validation
    # ============================================================

    def _validate_alpha(
        self,
        alpha: torch.Tensor,
        name: str,
    ) -> None:
        """
        Validate a Dirichlet concentration tensor.

        Expected shape:

            [B, K]

        with:

            alpha >= 1

        because HVNet constructs:

            alpha = softplus(logits) + 1.
        """

        if not torch.is_tensor(alpha):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if alpha.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, K], "
                f"but received {tuple(alpha.shape)}."
            )

        if alpha.size(1) != self.num_classes:
            raise ValueError(
                f"{name} contains {alpha.size(1)} classes, "
                f"but expected {self.num_classes}."
            )

        if not torch.isfinite(alpha).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        if torch.any(alpha < 1.0):
            raise ValueError(
                f"{name} must satisfy alpha >= 1 because "
                "HVNet defines alpha = softplus(logits) + 1."
            )

    # ============================================================
    # Evidence calibration
    # ============================================================

    def calibrate_alpha(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply evidence temperature calibration.

        Implements:

            alpha'_m =
                (alpha_m - 1) * tau + 1

        Parameters
        ----------
        alpha : torch.Tensor
            Original Dirichlet concentration parameters.
            Shape: [B, K].

        Returns
        -------
        torch.Tensor
            Calibrated Dirichlet concentration parameters.
            Shape: [B, K].
        """

        calibrated_alpha = (
            (alpha - 1.0) * self.temperature
            + 1.0
        )

        return calibrated_alpha

    # ============================================================
    # Dirichlet -> belief and uncertainty
    # ============================================================

    def dirichlet_to_opinion(
        self,
        alpha: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Convert calibrated Dirichlet parameters into belief masses
        and uncertainty.

        Implements:

            S_m = sum_k alpha'_{m,k}

            b_{m,k}
                = (alpha'_{m,k} - 1) / S_m

            u_m
                = K / S_m

        Parameters
        ----------
        alpha : torch.Tensor
            Calibrated Dirichlet concentration parameters.
            Shape: [B, K].

        Returns
        -------
        belief : torch.Tensor
            Belief masses b_m.
            Shape: [B, K].

        uncertainty : torch.Tensor
            Uncertainty u_m.
            Shape: [B, 1].

        strength : torch.Tensor
            Dirichlet strength S_m.
            Shape: [B, 1].

        evidence : torch.Tensor
            Calibrated evidence alpha' - 1.
            Shape: [B, K].
        """

        # --------------------------------------------------------
        # evidence_m = alpha'_m - 1
        # --------------------------------------------------------

        evidence = alpha - 1.0

        # --------------------------------------------------------
        # S_m = sum_k alpha'_{m,k}
        # --------------------------------------------------------

        strength = alpha.sum(
            dim=-1,
            keepdim=True,
        )

        strength = strength.clamp_min(
            self.eps
        )

        # --------------------------------------------------------
        # b_{m,k}
        #     = (alpha'_{m,k} - 1) / S_m
        # --------------------------------------------------------

        belief = (
            evidence
            / strength
        )

        # --------------------------------------------------------
        # u_m = K / S_m
        # --------------------------------------------------------

        uncertainty = (
            float(self.num_classes)
            / strength
        )

        return (
            belief,
            uncertainty,
            strength,
            evidence,
        )

    # ============================================================
    # Conflict calculation
    # ============================================================

    def compute_conflict(
        self,
        belief_clin: torch.Tensor,
        belief_vis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-class Dempster--Shafer conflict mass.

        Implements:

            C =
                sum_{i != j}
                b_{c,i} * b_{v,j}

        This can be computed efficiently as:

            C =
                (sum_i b_{c,i})
                (sum_j b_{v,j})
                -
                sum_k b_{c,k} b_{v,k}

        Parameters
        ----------
        belief_clin : torch.Tensor
            Clinical belief masses.
            Shape: [B, K].

        belief_vis : torch.Tensor
            Visual belief masses.
            Shape: [B, K].

        Returns
        -------
        torch.Tensor
            Conflict mass C.
            Shape: [B, 1].
        """

        clinical_total = belief_clin.sum(
            dim=-1,
            keepdim=True,
        )

        visual_total = belief_vis.sum(
            dim=-1,
            keepdim=True,
        )

        same_class_agreement = (
            belief_clin
            * belief_vis
        ).sum(
            dim=-1,
            keepdim=True,
        )

        conflict = (
            clinical_total
            * visual_total
            - same_class_agreement
        )

        # Small floating-point errors may occasionally generate
        # extremely small negative values.
        conflict = conflict.clamp_min(
            0.0
        )

        return conflict

    # ============================================================
    # DS fusion
    # ============================================================

    def forward(
        self,
        alpha_clin: torch.Tensor,
        alpha_vis: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse clinical and visual evidential opinions.

        Parameters
        ----------
        alpha_clin : torch.Tensor
            Clinical Dirichlet concentration parameters alpha_c.
            Shape: [B, K].

        alpha_vis : torch.Tensor
            Visual Dirichlet concentration parameters alpha_v.
            Shape: [B, K].

        Returns
        -------
        outputs : dict

            p_ds:
                DS fused predictive distribution.
                Shape: [B, K].

            b_ds:
                Fused belief masses.
                Shape: [B, K].

            u_ds:
                Fused uncertainty.
                Shape: [B, 1].

            b_clin:
                Clinical belief masses.
                Shape: [B, K].

            b_vis:
                Visual belief masses.
                Shape: [B, K].

            u_clin:
                Clinical uncertainty.
                Shape: [B, 1].

            u_vis:
                Visual uncertainty.
                Shape: [B, 1].

            conflict:
                Cross-modal conflict mass C.
                Shape: [B, 1].

            alpha_clin_calibrated:
                Temperature-calibrated clinical Dirichlet parameters.

            alpha_vis_calibrated:
                Temperature-calibrated visual Dirichlet parameters.
        """

        # --------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------

        self._validate_alpha(
            alpha_clin,
            name="alpha_clin",
        )

        self._validate_alpha(
            alpha_vis,
            name="alpha_vis",
        )

        if alpha_clin.size(0) != alpha_vis.size(0):
            raise ValueError(
                "alpha_clin and alpha_vis must have "
                "the same batch size."
            )

        # ========================================================
        # 1. Temperature calibration
        # ========================================================

        # --------------------------------------------------------
        # alpha'_c =
        #     (alpha_c - 1) * tau + 1
        # --------------------------------------------------------

        alpha_clin_calibrated = self.calibrate_alpha(
            alpha_clin
        )

        # --------------------------------------------------------
        # alpha'_v =
        #     (alpha_v - 1) * tau + 1
        # --------------------------------------------------------

        alpha_vis_calibrated = self.calibrate_alpha(
            alpha_vis
        )

        # ========================================================
        # 2. Convert Dirichlet parameters into opinions
        # ========================================================

        (
            belief_clin,
            uncertainty_clin,
            strength_clin,
            evidence_clin,
        ) = self.dirichlet_to_opinion(
            alpha_clin_calibrated
        )

        (
            belief_vis,
            uncertainty_vis,
            strength_vis,
            evidence_vis,
        ) = self.dirichlet_to_opinion(
            alpha_vis_calibrated
        )

        # ========================================================
        # 3. Unnormalized combined belief
        # ========================================================

        # --------------------------------------------------------
        # tilde_b_k =
        #
        #     b_{c,k} b_{v,k}
        #     +
        #     b_{c,k} u_v
        #     +
        #     u_c b_{v,k}
        # --------------------------------------------------------

        belief_unnormalized = (
            belief_clin * belief_vis
            + belief_clin * uncertainty_vis
            + uncertainty_clin * belief_vis
        )

        # ========================================================
        # 4. Cross-modal conflict
        # ========================================================

        # --------------------------------------------------------
        # C =
        #     sum_{i != j}
        #     b_{c,i} b_{v,j}
        # --------------------------------------------------------

        conflict = self.compute_conflict(
            belief_clin=belief_clin,
            belief_vis=belief_vis,
        )

        # ========================================================
        # 5. DS normalization
        # ========================================================

        # --------------------------------------------------------
        # denominator = 1 - C
        # --------------------------------------------------------

        normalization = (
            1.0 - conflict
        ).clamp_min(
            self.eps
        )

        # --------------------------------------------------------
        # b_ds,k =
        #     tilde_b_k / (1 - C)
        # --------------------------------------------------------

        belief_ds = (
            belief_unnormalized
            / normalization
        )

        # --------------------------------------------------------
        # u_ds =
        #     u_c * u_v / (1 - C)
        # --------------------------------------------------------

        uncertainty_ds = (
            uncertainty_clin
            * uncertainty_vis
            / normalization
        )

        # ========================================================
        # 6. DS predictive distribution
        # ========================================================

        # --------------------------------------------------------
        # p_ds,k =
        #
        #     b_ds,k
        #     -----------------
        #     sum_j b_ds,j
        # --------------------------------------------------------

        belief_sum = belief_ds.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(
            self.eps
        )

        p_ds = (
            belief_ds
            / belief_sum
        )

        # ========================================================
        # Outputs
        # ========================================================

        outputs = {
            # Final DS prediction
            "p_ds": p_ds,

            # Fused evidential opinion
            "b_ds": belief_ds,
            "u_ds": uncertainty_ds,

            # Modality-specific beliefs
            "b_clin": belief_clin,
            "b_vis": belief_vis,

            # These two values are subsequently used by the
            # sample-adaptive confidence gate:
            #
            # g = [h_clin; h_vis; u_c; u_v]
            "u_clin": uncertainty_clin,
            "u_vis": uncertainty_vis,

            # DS conflict
            "conflict": conflict,

            # Normalization term 1 - C
            "normalization": normalization,

            # Unnormalized combined belief
            "b_tilde": belief_unnormalized,

            # Temperature-calibrated Dirichlet parameters
            "alpha_clin_calibrated":
                alpha_clin_calibrated,

            "alpha_vis_calibrated":
                alpha_vis_calibrated,

            # Dirichlet strengths
            "strength_clin": strength_clin,
            "strength_vis": strength_vis,

            # Calibrated evidence
            "evidence_clin": evidence_clin,
            "evidence_vis": evidence_vis,
        }

        return outputs


# ============================================================
# Short aliases
# ============================================================

DSFusion = DempsterShaferFusion
DSEvidenceFusion = DempsterShaferFusion


# ============================================================
# Minimal sanity check
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 4
    num_classes = 3

    ds_fusion = DempsterShaferFusion(
        num_classes=num_classes,
        temperature=1.0,
    )

    # ------------------------------------------------------------
    # Example clinical and visual logits
    #
    # These imitate outputs from:
    #
    # clinical_encoder.py
    # mgva.py
    # ------------------------------------------------------------

    clinical_logits = torch.randn(
        batch_size,
        num_classes,
    )

    visual_logits = torch.randn(
        batch_size,
        num_classes,
    )

    # ------------------------------------------------------------
    # alpha_c = softplus(z_c) + 1
    # alpha_v = softplus(z_v) + 1
    # ------------------------------------------------------------

    alpha_clin = (
        torch.nn.functional.softplus(
            clinical_logits
        )
        + 1.0
    )

    alpha_vis = (
        torch.nn.functional.softplus(
            visual_logits
        )
        + 1.0
    )

    outputs = ds_fusion(
        alpha_clin=alpha_clin,
        alpha_vis=alpha_vis,
    )

    print(
        "Clinical alpha shape:",
        alpha_clin.shape,
    )

    print(
        "Visual alpha shape:",
        alpha_vis.shape,
    )

    print(
        "DS probability shape:",
        outputs["p_ds"].shape,
    )

    print(
        "\nClinical uncertainty:"
    )

    print(
        outputs["u_clin"]
    )

    print(
        "\nVisual uncertainty:"
    )

    print(
        outputs["u_vis"]
    )

    print(
        "\nConflict mass:"
    )

    print(
        outputs["conflict"]
    )

    print(
        "\nDS uncertainty:"
    )

    print(
        outputs["u_ds"]
    )

    print(
        "\nDS predictive distribution:"
    )

    print(
        outputs["p_ds"]
    )

    print(
        "\nDS probability sums:"
    )

    print(
        outputs["p_ds"].sum(dim=-1)
    )

    print(
        "\nDS opinion sums "
        "(sum belief + uncertainty):"
    )

    print(
        outputs["b_ds"].sum(
            dim=-1,
            keepdim=True,
        )
        + outputs["u_ds"]
    )

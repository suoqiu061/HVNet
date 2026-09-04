"""
Loss functions for HVNet.

This module implements the training objective described in the manuscript:

    L =
        CE(p_final, y)
        +
        lambda_aux * [
            CE(p_clin, y)
            +
            CE(p_vis, y)
        ]
        +
        lambda_edl * [
            KL_c
            +
            KL_v
        ]

where:

    p_final : final HVNet predictive distribution
    p_clin  : clinical-branch predictive distribution
    p_vis   : visual-branch predictive distribution

    KL_c    : Dirichlet evidential regularization for the clinical branch
    KL_v    : Dirichlet evidential regularization for the visual branch

The clinical and visual branches construct Dirichlet concentration
parameters as:

    alpha_clin = softplus(z_c) + 1

    alpha_vis  = softplus(z_v) + 1

The KL term regularizes the corresponding Dirichlet distribution toward
a uniform Dirichlet prior:

    Dir(1, ..., 1)

Unless the original experimental implementation used a different
Dirichlet KL formulation, this direct KL-to-uniform formulation should
be retained consistently across code and manuscript.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Probability-based cross entropy
# ======================================================================


def probability_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy loss for an already normalized predictive distribution.

    Implements:

        CE(p, y) = -log(p_y)

    This function should be used instead of nn.CrossEntropyLoss when
    the input is already a probability distribution produced by
    softmax.

    Parameters
    ----------
    probabilities : torch.Tensor
        Predictive probabilities.
        Shape: [B, K].

    targets : torch.Tensor
        Integer target labels.
        Shape: [B].

    eps : float
        Numerical stability constant.

    reduction : str
        Reduction mode:
            "mean"
            "sum"
            "none"

    Returns
    -------
    torch.Tensor
        Cross-entropy loss.
    """

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape [B, K], "
            f"but received {tuple(probabilities.shape)}."
        )

    if targets.ndim == 2 and targets.size(1) == 1:
        targets = targets.squeeze(1)

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape [B] or [B, 1], "
            f"but received {tuple(targets.shape)}."
        )

    if probabilities.size(0) != targets.size(0):
        raise ValueError(
            "probabilities and targets must have the same batch size."
        )

    targets = targets.long()

    num_classes = probabilities.size(1)

    if targets.numel() > 0:
        target_min = int(targets.min().item())
        target_max = int(targets.max().item())

        if target_min < 0 or target_max >= num_classes:
            raise ValueError(
                f"Target labels must lie in [0, {num_classes - 1}], "
                f"but observed [{target_min}, {target_max}]."
            )

    # Numerical stabilization.
    probabilities = probabilities.clamp(
        min=eps,
        max=1.0,
    )

    log_probabilities = torch.log(
        probabilities
    )

    # Equivalent to:
    #
    #   -log(p_y)
    #
    loss = F.nll_loss(
        log_probabilities,
        targets,
        reduction=reduction,
    )

    return loss


# ======================================================================
# Dirichlet KL divergence
# ======================================================================


def dirichlet_kl_divergence(
    alpha: torch.Tensor,
    prior_alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between two Dirichlet distributions.

    Implements:

        KL(
            Dir(alpha)
            ||
            Dir(prior_alpha)
        )

    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet concentration parameters.
        Shape: [B, K].

    prior_alpha : torch.Tensor
        Prior Dirichlet concentration parameters.

        Accepted shapes:
            [K]
            [1, K]
            [B, K]

    Returns
    -------
    torch.Tensor
        KL divergence for each sample.
        Shape: [B].
    """

    if alpha.ndim != 2:
        raise ValueError(
            "alpha must have shape [B, K]."
        )

    if prior_alpha.ndim == 1:
        prior_alpha = prior_alpha.unsqueeze(0)

    if prior_alpha.ndim != 2:
        raise ValueError(
            "prior_alpha must have shape [K], [1, K], or [B, K]."
        )

    if prior_alpha.size(1) != alpha.size(1):
        raise ValueError(
            "alpha and prior_alpha must contain the same "
            "number of classes."
        )

    if prior_alpha.size(0) == 1:
        prior_alpha = prior_alpha.expand(
            alpha.size(0),
            -1,
        )

    elif prior_alpha.size(0) != alpha.size(0):
        raise ValueError(
            "prior_alpha batch size must be either 1 "
            "or equal to alpha batch size."
        )

    if torch.any(alpha <= 0):
        raise ValueError(
            "All Dirichlet alpha values must be positive."
        )

    if torch.any(prior_alpha <= 0):
        raise ValueError(
            "All prior Dirichlet alpha values must be positive."
        )

    # --------------------------------------------------------------
    # Sum of concentration parameters
    # --------------------------------------------------------------

    alpha_0 = alpha.sum(
        dim=-1,
        keepdim=True,
    )

    prior_alpha_0 = prior_alpha.sum(
        dim=-1,
        keepdim=True,
    )

    # --------------------------------------------------------------
    # KL(Dir(alpha) || Dir(beta))
    #
    # log Gamma(sum alpha)
    # - sum log Gamma(alpha_k)
    #
    # - log Gamma(sum beta)
    # + sum log Gamma(beta_k)
    #
    # + sum_k [
    #     (alpha_k - beta_k)
    #     (
    #       digamma(alpha_k)
    #       - digamma(sum alpha)
    #     )
    #   ]
    # --------------------------------------------------------------

    log_normalizer_alpha = (
        torch.lgamma(alpha_0)
        - torch.lgamma(alpha).sum(
            dim=-1,
            keepdim=True,
        )
    )

    log_normalizer_prior = (
        torch.lgamma(prior_alpha_0)
        - torch.lgamma(prior_alpha).sum(
            dim=-1,
            keepdim=True,
        )
    )

    expectation_term = (
        (alpha - prior_alpha)
        * (
            torch.digamma(alpha)
            - torch.digamma(alpha_0)
        )
    ).sum(
        dim=-1,
        keepdim=True,
    )

    kl = (
        log_normalizer_alpha
        - log_normalizer_prior
        + expectation_term
    )

    return kl.squeeze(-1)


# ======================================================================
# Uniform-prior EDL regularization
# ======================================================================


def evidential_kl_loss(
    alpha: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Dirichlet evidential KL regularization.

    The prior is the uniform Dirichlet distribution:

        beta = [1, 1, ..., 1]

    Therefore:

        KL =
            KL(
                Dir(alpha)
                ||
                Dir(1)
            )

    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet concentration parameters.
        Shape: [B, K].

    reduction : str
        Reduction mode:
            "mean"
            "sum"
            "none"

    Returns
    -------
    torch.Tensor
        Evidential KL loss.
    """

    if alpha.ndim != 2:
        raise ValueError(
            "alpha must have shape [B, K]."
        )

    prior_alpha = torch.ones_like(
        alpha
    )

    kl = dirichlet_kl_divergence(
        alpha=alpha,
        prior_alpha=prior_alpha,
    )

    if reduction == "mean":
        return kl.mean()

    if reduction == "sum":
        return kl.sum()

    if reduction == "none":
        return kl

    raise ValueError(
        "reduction must be 'mean', 'sum', or 'none'."
    )


# ======================================================================
# Complete HVNet objective
# ======================================================================


class HVNetLoss(nn.Module):
    """
    Complete HVNet training objective.

    Implements:

        L =
            CE(p_final, y)

            + lambda_aux * [
                CE(p_clin, y)
                +
                CE(p_vis, y)
            ]

            + lambda_edl * [
                KL_c
                +
                KL_v
            ]

    Manuscript configuration:

        lambda_aux = 0.3
        lambda_edl = 0.1

    Parameters
    ----------
    lambda_aux : float
        Auxiliary classification loss weight.

    lambda_edl : float
        Dirichlet evidential KL regularization weight.

    eps : float
        Numerical stability constant for probability-based CE.
    """

    def __init__(
        self,
        lambda_aux: float = 0.3,
        lambda_edl: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if lambda_aux < 0:
            raise ValueError(
                "lambda_aux must be non-negative."
            )

        if lambda_edl < 0:
            raise ValueError(
                "lambda_edl must be non-negative."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be greater than zero."
            )

        self.lambda_aux = lambda_aux
        self.lambda_edl = lambda_edl
        self.eps = eps

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the complete HVNet loss.

        Parameters
        ----------
        outputs : dict
            Output dictionary returned by models/hvnet.py.

            Required entries:

                p_final
                p_clin
                p_vis
                alpha_clin
                alpha_vis

        targets : torch.Tensor
            Integer class labels.
            Shape: [B].

        Returns
        -------
        dict
            total_loss
                Complete HVNet training loss.

            loss_final
                Final prediction CE.

            loss_clin
                Clinical auxiliary CE.

            loss_vis
                Visual auxiliary CE.

            loss_aux
                Sum of the two auxiliary CEs.

            kl_clin
                Clinical evidential KL.

            kl_vis
                Visual evidential KL.

            loss_edl
                Sum of clinical and visual KL terms.

            weighted_aux
                lambda_aux * loss_aux.

            weighted_edl
                lambda_edl * loss_edl.
        """

        required_keys = [
            "p_final",
            "p_clin",
            "p_vis",
            "alpha_clin",
            "alpha_vis",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in outputs
        ]

        if missing_keys:
            raise KeyError(
                "HVNet outputs are missing required keys: "
                + ", ".join(missing_keys)
            )

        p_final = outputs["p_final"]
        p_clin = outputs["p_clin"]
        p_vis = outputs["p_vis"]

        alpha_clin = outputs["alpha_clin"]
        alpha_vis = outputs["alpha_vis"]

        # ==========================================================
        # 1. Primary classification term
        #
        # CE(p_final, y)
        # ==========================================================

        loss_final = probability_cross_entropy(
            probabilities=p_final,
            targets=targets,
            eps=self.eps,
            reduction="mean",
        )

        # ==========================================================
        # 2. Auxiliary branch classification terms
        #
        # CE(p_clin, y)
        # +
        # CE(p_vis, y)
        # ==========================================================

        loss_clin = probability_cross_entropy(
            probabilities=p_clin,
            targets=targets,
            eps=self.eps,
            reduction="mean",
        )

        loss_vis = probability_cross_entropy(
            probabilities=p_vis,
            targets=targets,
            eps=self.eps,
            reduction="mean",
        )

        loss_aux = (
            loss_clin
            + loss_vis
        )

        # ==========================================================
        # 3. Dirichlet evidential regularization
        #
        # KL^c + KL^v
        # ==========================================================

        kl_clin = evidential_kl_loss(
            alpha=alpha_clin,
            reduction="mean",
        )

        kl_vis = evidential_kl_loss(
            alpha=alpha_vis,
            reduction="mean",
        )

        loss_edl = (
            kl_clin
            + kl_vis
        )

        # ==========================================================
        # 4. Apply manuscript loss weights
        # ==========================================================

        weighted_aux = (
            self.lambda_aux
            * loss_aux
        )

        weighted_edl = (
            self.lambda_edl
            * loss_edl
        )

        # ==========================================================
        # 5. Overall objective
        #
        # L =
        #
        # CE(p_final, y)
        #
        # + lambda_aux [
        #       CE(p_clin, y)
        #       +
        #       CE(p_vis, y)
        #   ]
        #
        # + lambda_edl [
        #       KL_c
        #       +
        #       KL_v
        #   ]
        # ==========================================================

        total_loss = (
            loss_final
            + weighted_aux
            + weighted_edl
        )

        return {
            "total_loss": total_loss,

            "loss_final": loss_final,

            "loss_clin": loss_clin,
            "loss_vis": loss_vis,
            "loss_aux": loss_aux,

            "kl_clin": kl_clin,
            "kl_vis": kl_vis,
            "loss_edl": loss_edl,

            "weighted_aux": weighted_aux,
            "weighted_edl": weighted_edl,
        }


# ======================================================================
# Short alias
# ======================================================================

HVNetObjective = HVNetLoss


# ======================================================================
# Minimal sanity check
# ======================================================================


if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 4
    num_classes = 3

    # --------------------------------------------------------------
    # Example predictive distributions
    # --------------------------------------------------------------

    p_final = torch.softmax(
        torch.randn(
            batch_size,
            num_classes,
        ),
        dim=-1,
    )

    p_clin = torch.softmax(
        torch.randn(
            batch_size,
            num_classes,
        ),
        dim=-1,
    )

    p_vis = torch.softmax(
        torch.randn(
            batch_size,
            num_classes,
        ),
        dim=-1,
    )

    # --------------------------------------------------------------
    # Example Dirichlet parameters
    #
    # alpha = softplus(z) + 1
    # --------------------------------------------------------------

    logits_clin = torch.randn(
        batch_size,
        num_classes,
    )

    logits_vis = torch.randn(
        batch_size,
        num_classes,
    )

    alpha_clin = (
        F.softplus(
            logits_clin
        )
        + 1.0
    )

    alpha_vis = (
        F.softplus(
            logits_vis
        )
        + 1.0
    )

    targets = torch.tensor(
        [
            0,
            1,
            2,
            1,
        ],
        dtype=torch.long,
    )

    outputs = {
        "p_final": p_final,
        "p_clin": p_clin,
        "p_vis": p_vis,
        "alpha_clin": alpha_clin,
        "alpha_vis": alpha_vis,
    }

    criterion = HVNetLoss(
        lambda_aux=0.3,
        lambda_edl=0.1,
    )

    losses = criterion(
        outputs=outputs,
        targets=targets,
    )

    print(
        "Final CE:",
        losses["loss_final"].item(),
    )

    print(
        "Clinical CE:",
        losses["loss_clin"].item(),
    )

    print(
        "Visual CE:",
        losses["loss_vis"].item(),
    )

    print(
        "Clinical KL:",
        losses["kl_clin"].item(),
    )

    print(
        "Visual KL:",
        losses["kl_vis"].item(),
    )

    print(
        "Weighted auxiliary loss:",
        losses["weighted_aux"].item(),
    )

    print(
        "Weighted EDL loss:",
        losses["weighted_edl"].item(),
    )

    print(
        "Total HVNet loss:",
        losses["total_loss"].item(),
    )

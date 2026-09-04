"""
Clinical Hypothesis Encoder for HVNet.

This module implements the Hypothesize stage of HVNet:

    E = [Emb_1(i_1); Emb_2(i_2); Emb_3(i_3)]

    h_clin = W_c Flatten(TransformerEnc(E)) + b_c

    z_c     = f_c(h_clin)
    p_clin  = softmax(z_c)
    alpha_c = softplus(z_c) + 1

The three structured clinical indicators are:

    1. IOP: binary status
       0 = not elevated
       1 = elevated (> 21 mmHg)

    2. ACD: binary SLE-derived anterior chamber depth
       0 = normal
       1 = shallow

    3. CAG: Van Herick-based chamber-angle grade
       integer grade from 0 to 4

In the schematic figures and selected ablation settings of the manuscript,
this structured clinical branch may be labeled as "Text". It does NOT
represent free-text or natural-language input.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalHypothesisEncoder(nn.Module):
    """
    Structured clinical encoder used in the Hypothesize stage of HVNet.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
        HVNet uses three classes: Normal, OAG, and ACG.

    embedding_dim : int
        Embedding dimension for each structured clinical indicator.

    num_heads : int
        Number of attention heads in the Transformer encoder.

    num_layers : int
        Number of Transformer encoder layers.

    feedforward_dim : int
        Hidden dimension of the feed-forward network inside each
        Transformer encoder layer.

    clinical_feature_dim : int
        Output dimension of the clinical representation h_clin.

    dropout : float
        Dropout probability used inside the Transformer encoder.
    """

    NUM_IOP_STATES = 2
    NUM_ACD_STATES = 2
    NUM_CAG_STATES = 5
    NUM_CLINICAL_TOKENS = 3

    def __init__(
        self,
        num_classes: int = 3,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        clinical_feature_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "`embedding_dim` must be divisible by `num_heads`. "
                f"Received embedding_dim={embedding_dim}, "
                f"num_heads={num_heads}."
            )

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.clinical_feature_dim = clinical_feature_dim

        # ------------------------------------------------------------
        # Indicator embeddings
        #
        # Emb_1(i_1): IOP
        # Emb_2(i_2): ACD
        # Emb_3(i_3): CAG
        # ------------------------------------------------------------

        self.iop_embedding = nn.Embedding(
            num_embeddings=self.NUM_IOP_STATES,
            embedding_dim=embedding_dim,
        )

        self.acd_embedding = nn.Embedding(
            num_embeddings=self.NUM_ACD_STATES,
            embedding_dim=embedding_dim,
        )

        self.cag_embedding = nn.Embedding(
            num_embeddings=self.NUM_CAG_STATES,
            embedding_dim=embedding_dim,
        )

        # ------------------------------------------------------------
        # Transformer encoder
        #
        # Input:
        #     [B, 3, embedding_dim]
        #
        # Output:
        #     [B, 3, embedding_dim]
        # ------------------------------------------------------------

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_layers,
        )

        # ------------------------------------------------------------
        # Clinical representation
        #
        # h_clin =
        # W_c Flatten(TransformerEnc(E)) + b_c
        # ------------------------------------------------------------

        flattened_dim = self.NUM_CLINICAL_TOKENS * embedding_dim

        self.clinical_projection = nn.Linear(
            in_features=flattened_dim,
            out_features=clinical_feature_dim,
        )

        # ------------------------------------------------------------
        # Clinical classifier
        #
        # z_c = f_c(h_clin)
        # ------------------------------------------------------------

        self.classifier = nn.Linear(
            in_features=clinical_feature_dim,
            out_features=num_classes,
        )

    @staticmethod
    def _prepare_indicator(
        x: torch.Tensor,
        name: str,
        min_value: int,
        max_value: int,
    ) -> torch.Tensor:
        """
        Convert an indicator tensor into a 1-D LongTensor suitable for
        nn.Embedding and validate its discrete range.

        Accepted input shapes:
            [B]
            [B, 1]
        """

        if not torch.is_tensor(x):
            raise TypeError(
                f"{name} must be a torch.Tensor, "
                f"but received {type(x).__name__}."
            )

        if x.ndim == 2 and x.size(1) == 1:
            x = x.squeeze(1)

        if x.ndim != 1:
            raise ValueError(
                f"{name} must have shape [B] or [B, 1], "
                f"but received shape {tuple(x.shape)}."
            )

        x = x.long()

        if x.numel() > 0:
            x_min = int(x.min().item())
            x_max = int(x.max().item())

            if x_min < min_value or x_max > max_value:
                raise ValueError(
                    f"{name} contains invalid values. "
                    f"Expected integers in [{min_value}, {max_value}], "
                    f"but observed range [{x_min}, {x_max}]."
                )

        return x

    def forward(
        self,
        iop: torch.Tensor,
        acd: torch.Tensor,
        cag: torch.Tensor,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the clinical Hypothesize branch.

        Parameters
        ----------
        iop : torch.Tensor
            Binary IOP indicator.
            Shape: [B] or [B, 1].

        acd : torch.Tensor
            Binary SLE-derived ACD indicator.
            Shape: [B] or [B, 1].

        cag : torch.Tensor
            Van Herick CAG in the range 0--4.
            Shape: [B] or [B, 1].

        return_tokens : bool, optional
            If True, also return the Transformer output tokens.

        Returns
        -------
        outputs : dict
            Dictionary containing:

            h_clin:
                Clinical representation.
                Shape: [B, clinical_feature_dim].

            logits:
                Clinical logits z_c.
                Shape: [B, K].

            p_clin:
                Clinical predictive distribution.
                Shape: [B, K].

            evidence:
                Non-negative Dirichlet evidence.
                Shape: [B, K].

            alpha_clin:
                Dirichlet concentration parameters.
                Shape: [B, K].

            tokens:
                Transformer output tokens, returned only when
                return_tokens=True.
        """

        # ------------------------------------------------------------
        # 1. Validate and prepare structured indicators
        # ------------------------------------------------------------

        iop = self._prepare_indicator(
            iop,
            name="IOP",
            min_value=0,
            max_value=1,
        )

        acd = self._prepare_indicator(
            acd,
            name="ACD",
            min_value=0,
            max_value=1,
        )

        cag = self._prepare_indicator(
            cag,
            name="CAG",
            min_value=0,
            max_value=4,
        )

        batch_size = iop.size(0)

        if acd.size(0) != batch_size or cag.size(0) != batch_size:
            raise ValueError(
                "IOP, ACD, and CAG must have the same batch size."
            )

        # ------------------------------------------------------------
        # 2. Indicator embeddings
        #
        # E = [Emb_1(i_1); Emb_2(i_2); Emb_3(i_3)]
        # ------------------------------------------------------------

        iop_token = self.iop_embedding(iop)
        acd_token = self.acd_embedding(acd)
        cag_token = self.cag_embedding(cag)

        # [B, 3, D]
        clinical_tokens = torch.stack(
            [
                iop_token,
                acd_token,
                cag_token,
            ],
            dim=1,
        )

        # ------------------------------------------------------------
        # 3. Transformer encoding
        # ------------------------------------------------------------

        encoded_tokens = self.transformer_encoder(clinical_tokens)

        # ------------------------------------------------------------
        # 4. Flatten structured clinical tokens
        # ------------------------------------------------------------

        flattened = encoded_tokens.reshape(batch_size, -1)

        # ------------------------------------------------------------
        # 5. Clinical representation
        #
        # h_clin =
        # W_c Flatten(TransformerEnc(E)) + b_c
        # ------------------------------------------------------------

        h_clin = self.clinical_projection(flattened)

        # ------------------------------------------------------------
        # 6. Clinical prediction
        #
        # z_c = f_c(h_clin)
        # p_clin = softmax(z_c)
        # ------------------------------------------------------------

        logits = self.classifier(h_clin)

        p_clin = F.softmax(
            logits,
            dim=-1,
        )

        # ------------------------------------------------------------
        # 7. Dirichlet evidential representation
        #
        # evidence = softplus(z_c)
        # alpha_c  = evidence + 1
        # ------------------------------------------------------------

        evidence = F.softplus(logits)

        alpha_clin = evidence + 1.0

        outputs = {
            "h_clin": h_clin,
            "logits": logits,
            "p_clin": p_clin,
            "evidence": evidence,
            "alpha_clin": alpha_clin,
        }

        if return_tokens:
            outputs["tokens"] = encoded_tokens

        return outputs


# ----------------------------------------------------------------
# Optional alias
#
# Allows importing:
#
#     from models.clinical_encoder import ClinicalEncoder
#
# instead of the longer class name.
# ----------------------------------------------------------------

ClinicalEncoder = ClinicalHypothesisEncoder


if __name__ == "__main__":
    """
    Minimal sanity check.

    This block is executed only when running:

        python models/clinical_encoder.py

    It is not executed when the module is imported by HVNet.
    """

    torch.manual_seed(42)

    model = ClinicalHypothesisEncoder(
        num_classes=3,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
        clinical_feature_dim=256,
        dropout=0.1,
    )

    # Example batch of four samples.
    iop = torch.tensor([0, 1, 1, 0])
    acd = torch.tensor([0, 0, 1, 1])
    cag = torch.tensor([4, 3, 1, 0])

    outputs = model(
        iop=iop,
        acd=acd,
        cag=cag,
        return_tokens=True,
    )

    print("Clinical representation:", outputs["h_clin"].shape)
    print("Clinical logits:", outputs["logits"].shape)
    print("Clinical probabilities:", outputs["p_clin"].shape)
    print("Dirichlet alpha:", outputs["alpha_clin"].shape)
    print("Transformer tokens:", outputs["tokens"].shape)

    print("\nExample clinical probabilities:")
    print(outputs["p_clin"])

    print("\nProbability sums:")
    print(outputs["p_clin"].sum(dim=1))

"""
HVNet: Hypothesis-and-Verify Network.

This file implements the complete HVNet architecture described in the
manuscript:

    Glaucoma Subtyping via Hypothesis-Guided Verification
    and Arbitrated Fusion

HVNet integrates:

    1. One color fundus photograph (CFP)
    2. Three structured clinical indicators:
        - IOP
        - ACD
        - CAG

through the sequential:

    Hypothesize -> Verify -> Arbitrate

framework.

======================================================================
Overall formulation
======================================================================

Clinical branch
---------------

The structured clinical indicators are encoded to obtain:

    h_clin
    p_clin
    alpha_clin


Visual verification branch
--------------------------

The CFP is processed by a ConvNeXt-Tiny backbone to obtain intermediate
feature maps:

    C2
    C3

The Multi-Granularity Visual Attention (MGVA) module then performs:

    h_vis
    p_vis
    alpha_vis


Dempster--Shafer arbitration
----------------------------

Clinical and visual evidential opinions are fused to obtain:

    p_ds
    u_c
    u_v


Sample-adaptive confidence gating
---------------------------------

The gating input is:

    g = [h_clin; h_vis; u_c; u_v]

For the manuscript configuration:

    dim(h_clin) = 256
    dim(h_vis)  = 256

therefore:

    dim(g) = 256 + 256 + 1 + 1 = 514

The gate predicts:

    [w_clin, w_vis, w_ds]
        = softmax(MLP_gate(g))

and the final prediction is:

    p_final =
        w_clin * p_clin
        + w_vis * p_vis
        + w_ds * p_ds


Terminology
-----------

In schematic figures and selected ablation settings, the structured
clinical branch may be labeled as "Text".

"Text" refers to the combination of IOP, ACD, and CAG and does NOT
represent free-text or natural-language input.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "HVNet requires the `timm` package. "
        "Install it using: pip install timm"
    ) from exc


# ======================================================================
# Local HVNet modules
# ======================================================================

try:
    # Package-style imports:
    #
    #     from models.hvnet import HVNet
    #
    from .clinical_encoder import ClinicalHypothesisEncoder
    from .mgva import MultiGranularityVisualAttention
    from .ds_fusion import DempsterShaferFusion
    from .confidence_gate import SampleAdaptiveConfidenceGate

except ImportError:
    # Allows direct execution:
    #
    #     python models/hvnet.py
    #
    from clinical_encoder import ClinicalHypothesisEncoder
    from mgva import MultiGranularityVisualAttention
    from ds_fusion import DempsterShaferFusion
    from confidence_gate import SampleAdaptiveConfidenceGate


# ======================================================================
# HVNet
# ======================================================================


class HVNet(nn.Module):
    """
    Complete Hypothesis-and-Verify Network.

    Parameters
    ----------
    num_classes : int
        Number of target classes.

        Manuscript configuration:
            3

        corresponding to:
            Normal
            OAG
            ACG

    backbone_name : str
        Name of the timm visual backbone.

        Manuscript configuration:
            convnext_tiny

    pretrained : bool
        Whether to initialize the visual backbone with ImageNet-pretrained
        parameters.

    backbone_out_indices : tuple[int, int]
        Two backbone stages used as C2 and C3.

        These feature maps are passed to MGVA.

        IMPORTANT:
        The exact stage indices should match the original experimental
        implementation.

    clinical_feature_dim : int
        Dimension of h_clin.

        Manuscript-consistent configuration:
            256

    clinical_embedding_dim : int
        Embedding dimension used for each structured clinical indicator.

    clinical_num_heads : int
        Number of Transformer heads in the clinical encoder.

    clinical_num_layers : int
        Number of clinical Transformer encoder layers.

    clinical_feedforward_dim : int
        Feed-forward dimension inside the clinical Transformer.

    clinical_dropout : float
        Dropout probability in the clinical encoder.

    attention_dim : int
        Global cross-attention dimension in MGVA.

    global_dim : int
        Dimension of h_global.

    local_dim : int
        Dimension of local-region representations.

    visual_feature_dim : int
        Dimension of h_vis.

        Manuscript-consistent configuration:
            256

    visual_attention_heads : int
        Number of heads in global MGVA attention.

    num_local_patches : int
        Number of local salient regions K.

        Manuscript configuration:
            K = 4

    patch_size : int
        Spatial size of each sampled local patch.

        This implementation detail should be matched to the original
        experimental code.

    local_transformer_heads : int
        Number of attention heads in the local-region Transformer.

    local_transformer_layers : int
        Number of local Transformer encoder layers.

    local_feedforward_dim : int
        Feed-forward dimension of the local Transformer.

    visual_dropout : float
        Dropout used in MGVA.

    ds_temperature : float
        Dempster--Shafer evidence calibration temperature tau.

        Manuscript configuration:
            tau = 1.0

    gate_hidden_dim : int
        Hidden dimension d_gate of the confidence gating MLP.

        Corresponds to:

            514 -> d_gate -> 3

    gate_dropout : float
        Dropout in the confidence gate.
    """

    def __init__(
        self,
        num_classes: int = 3,

        # ----------------------------------------------------------
        # Visual backbone
        # ----------------------------------------------------------
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        backbone_out_indices: Tuple[int, int] = (1, 2),

        # ----------------------------------------------------------
        # Clinical Hypothesis Encoder
        # ----------------------------------------------------------
        clinical_feature_dim: int = 256,
        clinical_embedding_dim: int = 64,
        clinical_num_heads: int = 4,
        clinical_num_layers: int = 2,
        clinical_feedforward_dim: int = 128,
        clinical_dropout: float = 0.1,

        # ----------------------------------------------------------
        # MGVA
        # ----------------------------------------------------------
        attention_dim: int = 256,
        global_dim: int = 256,
        local_dim: int = 256,
        visual_feature_dim: int = 256,
        visual_attention_heads: int = 4,
        num_local_patches: int = 4,
        patch_size: int = 3,
        local_transformer_heads: int = 4,
        local_transformer_layers: int = 1,
        local_feedforward_dim: int = 512,
        visual_dropout: float = 0.1,

        # ----------------------------------------------------------
        # DS fusion
        # ----------------------------------------------------------
        ds_temperature: float = 1.0,

        # ----------------------------------------------------------
        # Confidence gate
        # ----------------------------------------------------------
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.0,
    ) -> None:

        super().__init__()

        # ==========================================================
        # Basic configuration
        # ==========================================================

        self.num_classes = num_classes
        self.backbone_name = backbone_name

        self.clinical_feature_dim = clinical_feature_dim
        self.visual_feature_dim = visual_feature_dim

        if len(backbone_out_indices) != 2:
            raise ValueError(
                "backbone_out_indices must contain exactly two "
                "stage indices corresponding to C2 and C3."
            )

        # ==========================================================
        # 1. Clinical Hypothesis Encoder
        #
        # Hypothesize stage
        # ==========================================================

        self.clinical_encoder = ClinicalHypothesisEncoder(
            num_classes=num_classes,
            embedding_dim=clinical_embedding_dim,
            num_heads=clinical_num_heads,
            num_layers=clinical_num_layers,
            feedforward_dim=clinical_feedforward_dim,
            clinical_feature_dim=clinical_feature_dim,
            dropout=clinical_dropout,
        )

        # ==========================================================
        # 2. ConvNeXt-Tiny visual backbone
        # ==========================================================

        # features_only=True makes timm return intermediate feature maps
        # rather than the final ImageNet classification output.
        #
        # For two selected stages:
        #
        #     backbone(image) -> [C2, C3]
        #

        self.visual_backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=backbone_out_indices,
        )

        # ----------------------------------------------------------
        # Automatically determine the channel dimensions of
        # the selected backbone feature maps.
        # ----------------------------------------------------------

        backbone_channels = self.visual_backbone.feature_info.channels()

        if len(backbone_channels) != 2:
            raise RuntimeError(
                "The visual backbone must return exactly two "
                "feature maps for C2 and C3."
            )

        self.c2_channels = backbone_channels[0]
        self.c3_channels = backbone_channels[1]

        # ==========================================================
        # 3. Multi-Granularity Visual Attention
        #
        # Verify stage
        # ==========================================================

        self.mgva = MultiGranularityVisualAttention(
            c2_channels=self.c2_channels,
            c3_channels=self.c3_channels,
            clinical_dim=clinical_feature_dim,
            num_classes=num_classes,

            attention_dim=attention_dim,
            global_dim=global_dim,
            local_dim=local_dim,
            visual_feature_dim=visual_feature_dim,

            num_attention_heads=visual_attention_heads,

            num_local_patches=num_local_patches,
            patch_size=patch_size,

            local_transformer_heads=local_transformer_heads,
            local_transformer_layers=local_transformer_layers,
            local_feedforward_dim=local_feedforward_dim,

            dropout=visual_dropout,
        )

        # ==========================================================
        # 4. Dempster--Shafer Evidence Fusion
        #
        # Arbitrate stage - part 1
        # ==========================================================

        self.ds_fusion = DempsterShaferFusion(
            num_classes=num_classes,
            temperature=ds_temperature,
        )

        # ==========================================================
        # 5. Sample-Adaptive Confidence Gate
        #
        # Arbitrate stage - part 2
        # ==========================================================

        self.confidence_gate = SampleAdaptiveConfidenceGate(
            clinical_dim=clinical_feature_dim,
            visual_dim=visual_feature_dim,
            gate_hidden_dim=gate_hidden_dim,
            dropout=gate_dropout,
            activation="gelu",
        )

    # ==================================================================
    # Visual feature extraction
    # ==================================================================

    def extract_visual_features(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract C2 and C3 from the visual backbone.

        Parameters
        ----------
        image : torch.Tensor
            Preprocessed CFP image.

            Expected shape:
                [B, 3, H, W]

            Manuscript configuration:
                H = W = 224

        Returns
        -------
        C2 : torch.Tensor
            First selected intermediate visual feature map.

        C3 : torch.Tensor
            Second selected intermediate visual feature map.
        """

        if image.ndim != 4:
            raise ValueError(
                "image must have shape [B, 3, H, W], "
                f"but received {tuple(image.shape)}."
            )

        if image.size(1) != 3:
            raise ValueError(
                "HVNet expects RGB CFP images with 3 channels."
            )

        features = self.visual_backbone(
            image
        )

        if len(features) != 2:
            raise RuntimeError(
                "Expected the backbone to return exactly "
                "two feature maps."
            )

        c2, c3 = features

        return c2, c3

    # ==================================================================
    # Backbone freezing
    # ==================================================================

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters of the visual backbone.

        The manuscript training protocol freezes the backbone during
        the first five epochs.
        """

        for parameter in self.visual_backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all parameters of the visual backbone.

        Used after the initial backbone-freezing stage.
        """

        for parameter in self.visual_backbone.parameters():
            parameter.requires_grad = True

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        image: torch.Tensor,
        iop: torch.Tensor,
        acd: torch.Tensor,
        cag: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete HVNet forward pass.

        Parameters
        ----------
        image : torch.Tensor
            CFP image.
            Shape: [B, 3, H, W].

        iop : torch.Tensor
            Binary IOP indicator.
            Shape: [B] or [B, 1].

        acd : torch.Tensor
            Binary SLE-derived ACD indicator.
            Shape: [B] or [B, 1].

        cag : torch.Tensor
            Van Herick CAG.
            Integer values from 0 to 4.
            Shape: [B] or [B, 1].

        return_auxiliary : bool
            If True, return intermediate features and attention
            information in addition to the main predictions.

        Returns
        -------
        outputs : dict

            p_final
                Final HVNet predictive distribution.

            p_clin
                Clinical predictive distribution.

            p_vis
                Visual predictive distribution.

            p_ds
                Dempster--Shafer fused predictive distribution.

            alpha_clin
                Clinical Dirichlet concentration parameters.

            alpha_vis
                Visual Dirichlet concentration parameters.

            h_clin
                Clinical representation.

            h_vis
                Visual representation.

            u_clin
                Clinical evidential uncertainty.

            u_vis
                Visual evidential uncertainty.

            conflict
                Cross-modal Dempster--Shafer conflict.

            weights
                Confidence-gate weights:
                [w_clin, w_vis, w_ds].
        """

        # ==========================================================
        # Stage 1: HYPOTHESIZE
        # ==========================================================

        clinical_outputs = self.clinical_encoder(
            iop=iop,
            acd=acd,
            cag=cag,
            return_tokens=return_auxiliary,
        )

        h_clin = clinical_outputs["h_clin"]
        p_clin = clinical_outputs["p_clin"]
        alpha_clin = clinical_outputs["alpha_clin"]

        # ==========================================================
        # Visual backbone
        # ==========================================================

        c2, c3 = self.extract_visual_features(
            image
        )

        # ==========================================================
        # Stage 2: VERIFY
        #
        # Clinical hypothesis h_clin guides visual extraction.
        # ==========================================================

        visual_outputs = self.mgva(
            c2=c2,
            c3=c3,
            h_clin=h_clin,
            return_auxiliary=return_auxiliary,
        )

        h_vis = visual_outputs["h_vis"]
        p_vis = visual_outputs["p_vis"]
        alpha_vis = visual_outputs["alpha_vis"]

        # ==========================================================
        # Stage 3A: ARBITRATE
        #
        # Dempster--Shafer evidence fusion
        # ==========================================================

        ds_outputs = self.ds_fusion(
            alpha_clin=alpha_clin,
            alpha_vis=alpha_vis,
        )

        p_ds = ds_outputs["p_ds"]

        u_clin = ds_outputs["u_clin"]
        u_vis = ds_outputs["u_vis"]

        # ==========================================================
        # Stage 3B: ARBITRATE
        #
        # Sample-adaptive confidence gating
        # ==========================================================

        gate_outputs = self.confidence_gate(
            h_clin=h_clin,
            h_vis=h_vis,

            u_clin=u_clin,
            u_vis=u_vis,

            p_clin=p_clin,
            p_vis=p_vis,
            p_ds=p_ds,
        )

        # ----------------------------------------------------------
        # Final prediction:
        #
        # p_final =
        #     w_clin * p_clin
        #     +
        #     w_vis * p_vis
        #     +
        #     w_ds * p_ds
        # ----------------------------------------------------------

        p_final = gate_outputs["p_final"]

        # ==========================================================
        # Main outputs
        # ==========================================================

        outputs = {
            # ------------------------------------------------------
            # Final prediction
            # ------------------------------------------------------
            "p_final": p_final,

            # ------------------------------------------------------
            # Branch predictions
            # ------------------------------------------------------
            "p_clin": p_clin,
            "p_vis": p_vis,
            "p_ds": p_ds,

            # ------------------------------------------------------
            # Branch logits
            # ------------------------------------------------------
            "logits_clin": clinical_outputs["logits"],
            "logits_vis": visual_outputs["logits"],

            # ------------------------------------------------------
            # Dirichlet representations
            # ------------------------------------------------------
            "alpha_clin": alpha_clin,
            "alpha_vis": alpha_vis,

            # ------------------------------------------------------
            # Latent representations
            # ------------------------------------------------------
            "h_clin": h_clin,
            "h_vis": h_vis,

            # ------------------------------------------------------
            # Evidential uncertainty
            # ------------------------------------------------------
            "u_clin": u_clin,
            "u_vis": u_vis,

            # ------------------------------------------------------
            # DS conflict
            # ------------------------------------------------------
            "conflict": ds_outputs["conflict"],

            # ------------------------------------------------------
            # Confidence weights
            # ------------------------------------------------------
            "weights": gate_outputs["weights"],
            "w_clin": gate_outputs["w_clin"],
            "w_vis": gate_outputs["w_vis"],
            "w_ds": gate_outputs["w_ds"],
        }

        # ==========================================================
        # Optional detailed outputs
        # ==========================================================

        if return_auxiliary:

            outputs.update(
                {
                    # ------------------------------------------------
                    # Backbone feature maps
                    # ------------------------------------------------
                    "C2": c2,
                    "C3": c3,

                    # ------------------------------------------------
                    # Clinical tokens
                    # ------------------------------------------------
                    "clinical_tokens":
                        clinical_outputs.get("tokens"),

                    # ------------------------------------------------
                    # MGVA global/local representations
                    # ------------------------------------------------
                    "h_global":
                        visual_outputs["h_global"],

                    "h_local":
                        visual_outputs["h_local"],

                    # ------------------------------------------------
                    # Saliency-guided local verification
                    # ------------------------------------------------
                    "saliency_map":
                        visual_outputs.get("saliency_map"),

                    "topk_scores":
                        visual_outputs.get("topk_scores"),

                    "topk_indices":
                        visual_outputs.get("topk_indices"),

                    "local_patches":
                        visual_outputs.get("local_patches"),

                    "local_tokens":
                        visual_outputs.get("local_tokens"),

                    # ------------------------------------------------
                    # Global hypothesis-guided attention
                    # ------------------------------------------------
                    "attention_c2":
                        visual_outputs.get("attention_c2"),

                    "attention_c3":
                        visual_outputs.get("attention_c3"),

                    # ------------------------------------------------
                    # DS details
                    # ------------------------------------------------
                    "b_clin":
                        ds_outputs["b_clin"],

                    "b_vis":
                        ds_outputs["b_vis"],

                    "b_ds":
                        ds_outputs["b_ds"],

                    "u_ds":
                        ds_outputs["u_ds"],

                    "b_tilde":
                        ds_outputs["b_tilde"],

                    "ds_normalization":
                        ds_outputs["normalization"],

                    # ------------------------------------------------
                    # Confidence gate internals
                    # ------------------------------------------------
                    "gate_input":
                        gate_outputs["gate_input"],

                    "gate_logits":
                        gate_outputs["gate_logits"],
                }
            )

        return outputs


# ======================================================================
# Factory function
# ======================================================================


def build_hvnet(
    pretrained: bool = True,
    **kwargs,
) -> HVNet:
    """
    Convenience factory for constructing HVNet.

    Example
    -------

    model = build_hvnet(
        pretrained=True,
        num_classes=3,
    )
    """

    return HVNet(
        pretrained=pretrained,
        **kwargs,
    )


# ======================================================================
# Minimal sanity check
# ======================================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 2

    # --------------------------------------------------------------
    # Set pretrained=False for the sanity check so that running the
    # file does not require downloading ImageNet weights.
    # --------------------------------------------------------------

    model = HVNet(
        num_classes=3,
        backbone_name="convnext_tiny",
        pretrained=False,

        clinical_feature_dim=256,
        visual_feature_dim=256,

        num_local_patches=4,

        ds_temperature=1.0,
    )

    model.eval()

    # --------------------------------------------------------------
    # Example CFP batch
    #
    # In actual experiments these images should already have been
    # resized to 224 x 224 and ImageNet-normalized by the data loader.
    # --------------------------------------------------------------

    images = torch.randn(
        batch_size,
        3,
        224,
        224,
    )

    # --------------------------------------------------------------
    # Structured clinical indicators
    # --------------------------------------------------------------

    iop = torch.tensor(
        [0, 1]
    )

    acd = torch.tensor(
        [0, 1]
    )

    cag = torch.tensor(
        [4, 1]
    )

    with torch.no_grad():

        outputs = model(
            image=images,
            iop=iop,
            acd=acd,
            cag=cag,
            return_auxiliary=False,
        )

    print(
        "C2 channels:",
        model.c2_channels,
    )

    print(
        "C3 channels:",
        model.c3_channels,
    )

    print(
        "\nh_clin shape:",
        outputs["h_clin"].shape,
    )

    print(
        "h_vis shape:",
        outputs["h_vis"].shape,
    )

    print(
        "\np_clin shape:",
        outputs["p_clin"].shape,
    )

    print(
        "p_vis shape:",
        outputs["p_vis"].shape,
    )

    print(
        "p_ds shape:",
        outputs["p_ds"].shape,
    )

    print(
        "p_final shape:",
        outputs["p_final"].shape,
    )

    print(
        "\nConfidence weights:"
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
        "\nFinal predictions:"
    )

    print(
        outputs["p_final"]
    )

    print(
        "\nFinal probability sums:"
    )

    print(
        outputs["p_final"].sum(dim=-1)
    )

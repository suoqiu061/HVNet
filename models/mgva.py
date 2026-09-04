"""
Multi-Granularity Visual Attention (MGVA) module for HVNet.

This module implements the Verify stage of HVNet.

The implementation follows the formulation described in the manuscript:

Global verification branch
--------------------------

    T_1 = Flatten(C_2)
    T_2 = Flatten(C_3)

    o_s = MHA(phi_s(h_clin), T_s, T_s),    s in {1, 2}

    h_global = psi([o_1; o_2])


Local verification branch
-------------------------

    S = Conv_1x1(C_3)

    {P_i}_{i=1}^{K} = SampleTopK(C_2, S)

    r_i = rho(Flatten(P_i))

    h_local =
        [TransformerEnc([r_agg; r_1; ...; r_K])]_0


Visual representation and prediction
------------------------------------

    h_vis = MLP_vis([h_global; h_local])

    z_v = f_v(h_vis)

    p_vis = softmax(z_v)

    alpha_v = softplus(z_v) + 1


Notes
-----
1. h_clin is produced by the Clinical Hypothesis Encoder.

2. C_2 and C_3 are intermediate visual feature maps produced by
   the visual backbone.

3. K = 4 local salient regions in the manuscript configuration.

4. The structured clinical branch may be labeled "Text" in schematic
   figures. It does not represent free-text input.

5. Architectural dimensions that are not explicitly fixed by the
   manuscript equations are exposed as constructor arguments and
   should be set to the values used in the original experiment.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Top-K Salient Patch Sampling
# ============================================================


class TopKPatchSampler(nn.Module):
    """
    Sample K salient local regions from C2 according to a saliency map S.

    This module implements:

        {P_i}_{i=1}^{K} = SampleTopK(C_2, S)

    The saliency map generated from C3 is first resized to the spatial
    resolution of C2. The K highest-response spatial locations are then
    selected, and fixed-size local patches centered at these locations
    are extracted from C2.

    Parameters
    ----------
    num_patches : int
        Number of local regions K.

    patch_size : int
        Spatial size of each sampled local patch.

        An odd value is recommended so that the selected location
        corresponds naturally to the patch center.
    """

    def __init__(
        self,
        num_patches: int = 4,
        patch_size: int = 3,
    ) -> None:
        super().__init__()

        if num_patches <= 0:
            raise ValueError("num_patches must be greater than zero.")

        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero.")

        if patch_size % 2 == 0:
            raise ValueError(
                "An odd patch_size is recommended and required by this "
                "implementation so that the selected position is the "
                "center of the sampled patch."
            )

        self.num_patches = num_patches
        self.patch_size = patch_size

    def forward(
        self,
        feature_map: torch.Tensor,
        saliency_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feature_map : torch.Tensor
            C2 feature map.
            Shape: [B, C2, H2, W2]

        saliency_map : torch.Tensor
            Saliency map S generated from C3.
            Shape: [B, 1, H3, W3]

        Returns
        -------
        patches : torch.Tensor
            Sampled local patches.
            Shape:
                [B, K, C2, patch_size, patch_size]

        topk_scores : torch.Tensor
            Saliency scores of selected locations.
            Shape: [B, K]

        topk_indices : torch.Tensor
            Flattened spatial indices of selected locations on C2.
            Shape: [B, K]
        """

        if feature_map.ndim != 4:
            raise ValueError(
                "feature_map must have shape [B, C, H, W]."
            )

        if saliency_map.ndim != 4:
            raise ValueError(
                "saliency_map must have shape [B, 1, H, W]."
            )

        if saliency_map.size(1) != 1:
            raise ValueError(
                "saliency_map must contain exactly one saliency channel."
            )

        batch_size, channels, height, width = feature_map.shape

        if saliency_map.size(0) != batch_size:
            raise ValueError(
                "feature_map and saliency_map must have the same "
                "batch size."
            )

        # --------------------------------------------------------
        # Resize S from the C3 spatial scale to the C2 scale.
        # --------------------------------------------------------

        resized_saliency = F.interpolate(
            saliency_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # [B, H2 * W2]
        flattened_saliency = resized_saliency.flatten(start_dim=1)

        total_positions = height * width

        if self.num_patches > total_positions:
            raise ValueError(
                f"Requested K={self.num_patches} patches, but C2 "
                f"contains only {total_positions} spatial positions."
            )

        # --------------------------------------------------------
        # Select K highest-response locations.
        # --------------------------------------------------------

        topk_scores, topk_indices = torch.topk(
            flattened_saliency,
            k=self.num_patches,
            dim=1,
            largest=True,
            sorted=True,
        )

        # --------------------------------------------------------
        # Extract a patch centered at every C2 spatial position.
        #
        # unfold output:
        # [B, C2 * patch_size * patch_size, H2 * W2]
        # --------------------------------------------------------

        padding = self.patch_size // 2

        unfolded = F.unfold(
            feature_map,
            kernel_size=self.patch_size,
            padding=padding,
            stride=1,
        )

        # --------------------------------------------------------
        # Gather the K patches corresponding to the selected
        # saliency locations.
        # --------------------------------------------------------

        gather_indices = topk_indices.unsqueeze(1).expand(
            -1,
            unfolded.size(1),
            -1,
        )

        # [B, C2 * P * P, K]
        selected = torch.gather(
            unfolded,
            dim=2,
            index=gather_indices,
        )

        # [B, K, C2 * P * P]
        selected = selected.transpose(1, 2).contiguous()

        # [B, K, C2, P, P]
        patches = selected.view(
            batch_size,
            self.num_patches,
            channels,
            self.patch_size,
            self.patch_size,
        )

        return patches, topk_scores, topk_indices


# ============================================================
# Multi-Granularity Visual Attention
# ============================================================


class MultiGranularityVisualAttention(nn.Module):
    """
    Multi-Granularity Visual Attention (MGVA) module.

    The module contains:

        1. Hypothesis-guided global cross-attention;
        2. Saliency-guided local region modeling;
        3. Global-local visual representation fusion;
        4. Visual classification and evidential prediction.

    Parameters
    ----------
    c2_channels : int
        Number of channels in backbone feature map C2.

    c3_channels : int
        Number of channels in backbone feature map C3.

    clinical_dim : int
        Dimension of h_clin produced by the clinical encoder.

    num_classes : int
        Number of output classes. HVNet uses K_class = 3.

    attention_dim : int
        Embedding dimension used by global multi-head attention.

    global_dim : int
        Dimension of h_global.

    local_dim : int
        Dimension of local region tokens and h_local.

    visual_feature_dim : int
        Dimension of the final visual representation h_vis.

    num_attention_heads : int
        Number of heads in global multi-head attention.

    num_local_patches : int
        Number of salient local regions.
        Manuscript configuration: K = 4.

    patch_size : int
        Spatial size of each local patch P_i.

    local_transformer_heads : int
        Number of heads in the local Transformer encoder.

    local_transformer_layers : int
        Number of local Transformer encoder layers.

    local_feedforward_dim : int
        Feed-forward dimension in the local Transformer.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        c2_channels: int,
        c3_channels: int,
        clinical_dim: int,
        num_classes: int = 3,
        attention_dim: int = 256,
        global_dim: int = 256,
        local_dim: int = 256,
        visual_feature_dim: int = 256,
        num_attention_heads: int = 4,
        num_local_patches: int = 4,
        patch_size: int = 3,
        local_transformer_heads: int = 4,
        local_transformer_layers: int = 1,
        local_feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------

        if attention_dim % num_attention_heads != 0:
            raise ValueError(
                "attention_dim must be divisible by "
                "num_attention_heads."
            )

        if local_dim % local_transformer_heads != 0:
            raise ValueError(
                "local_dim must be divisible by "
                "local_transformer_heads."
            )

        self.c2_channels = c2_channels
        self.c3_channels = c3_channels
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.visual_feature_dim = visual_feature_dim
        self.num_local_patches = num_local_patches
        self.patch_size = patch_size

        # ========================================================
        # Global Verification Branch
        # ========================================================

        # --------------------------------------------------------
        # phi_1(h_clin) and phi_2(h_clin)
        #
        # The same clinical representation is independently
        # projected for the two visual scales.
        # --------------------------------------------------------

        self.phi_1 = nn.Linear(
            clinical_dim,
            attention_dim,
        )

        self.phi_2 = nn.Linear(
            clinical_dim,
            attention_dim,
        )

        # --------------------------------------------------------
        # o_s = MHA(phi_s(h_clin), T_s, T_s)
        #
        # PyTorch MultiheadAttention internally performs the
        # required key/value projections.
        #
        # kdim and vdim allow the raw flattened C2/C3 channel
        # dimensions to differ from attention_dim.
        # --------------------------------------------------------

        self.global_attention_c2 = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
            kdim=c2_channels,
            vdim=c2_channels,
        )

        self.global_attention_c3 = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
            kdim=c3_channels,
            vdim=c3_channels,
        )

        # --------------------------------------------------------
        # psi([o_1; o_2])
        #
        # Learnable projection from concatenated multi-scale
        # global responses to h_global.
        # --------------------------------------------------------

        self.psi = nn.Sequential(
            nn.Linear(
                attention_dim * 2,
                global_dim,
            ),
            nn.GELU(),
        )

        # ========================================================
        # Local Verification Branch
        # ========================================================

        # --------------------------------------------------------
        # S = Conv_1x1(C3)
        # --------------------------------------------------------

        self.saliency_head = nn.Conv2d(
            in_channels=c3_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # --------------------------------------------------------
        # {P_i}_{i=1}^{K} = SampleTopK(C2, S)
        # --------------------------------------------------------

        self.patch_sampler = TopKPatchSampler(
            num_patches=num_local_patches,
            patch_size=patch_size,
        )

        # --------------------------------------------------------
        # r_i = rho(Flatten(P_i))
        # --------------------------------------------------------

        flattened_patch_dim = (
            c2_channels * patch_size * patch_size
        )

        self.rho = nn.Sequential(
            nn.Linear(
                flattened_patch_dim,
                local_dim,
            ),
            nn.GELU(),
        )

        # --------------------------------------------------------
        # r_agg
        #
        # Learnable aggregation token placed before the K local
        # region tokens:
        #
        # [r_agg; r_1; ...; r_K]
        # --------------------------------------------------------

        self.r_agg = nn.Parameter(
            torch.zeros(
                1,
                1,
                local_dim,
            )
        )

        nn.init.trunc_normal_(
            self.r_agg,
            std=0.02,
        )

        # --------------------------------------------------------
        # TransformerEnc([r_agg; r_1; ...; r_K])
        # --------------------------------------------------------

        local_transformer_layer = nn.TransformerEncoderLayer(
            d_model=local_dim,
            nhead=local_transformer_heads,
            dim_feedforward=local_feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.local_transformer = nn.TransformerEncoder(
            encoder_layer=local_transformer_layer,
            num_layers=local_transformer_layers,
        )

        # ========================================================
        # Global-Local Visual Fusion
        # ========================================================

        # --------------------------------------------------------
        # h_vis = MLP_vis([h_global; h_local])
        # --------------------------------------------------------

        self.visual_mlp = nn.Sequential(
            nn.Linear(
                global_dim + local_dim,
                visual_feature_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                visual_feature_dim,
                visual_feature_dim,
            ),
        )

        # ========================================================
        # Visual Classifier
        # ========================================================

        # --------------------------------------------------------
        # z_v = f_v(h_vis)
        # --------------------------------------------------------

        self.visual_classifier = nn.Linear(
            visual_feature_dim,
            num_classes,
        )

    @staticmethod
    def _flatten_feature_map(
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flatten a visual feature map into spatial tokens.

        Implements:

            T_s = Flatten(C_s)

        Input
        -----
        [B, C, H, W]

        Output
        ------
        [B, H*W, C]
        """

        if feature_map.ndim != 4:
            raise ValueError(
                "Visual feature map must have shape [B, C, H, W]."
            )

        return (
            feature_map
            .flatten(start_dim=2)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        h_clin: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MGVA Verify stage.

        Parameters
        ----------
        c2 : torch.Tensor
            Intermediate backbone feature map C2.
            Shape: [B, C2_channels, H2, W2].

        c3 : torch.Tensor
            Higher-level backbone feature map C3.
            Shape: [B, C3_channels, H3, W3].

        h_clin : torch.Tensor
            Clinical hypothesis representation produced by the
            Clinical Hypothesis Encoder.
            Shape: [B, clinical_dim].

        return_auxiliary : bool
            If True, additionally return saliency maps, selected
            locations, local tokens, and global attention weights.

        Returns
        -------
        outputs : dict

            h_global:
                Global hypothesis-guided visual representation.

            h_local:
                Local salient-region representation.

            h_vis:
                Final visual representation.

            logits:
                Visual logits z_v.

            p_vis:
                Visual predictive distribution.

            evidence:
                Non-negative visual evidence.

            alpha_vis:
                Visual Dirichlet concentration parameters.
        """

        # --------------------------------------------------------
        # Validate input shapes
        # --------------------------------------------------------

        if c2.ndim != 4:
            raise ValueError(
                f"C2 must have shape [B, C, H, W], "
                f"received {tuple(c2.shape)}."
            )

        if c3.ndim != 4:
            raise ValueError(
                f"C3 must have shape [B, C, H, W], "
                f"received {tuple(c3.shape)}."
            )

        if h_clin.ndim != 2:
            raise ValueError(
                f"h_clin must have shape [B, clinical_dim], "
                f"received {tuple(h_clin.shape)}."
            )

        batch_size = c2.size(0)

        if c3.size(0) != batch_size:
            raise ValueError(
                "C2 and C3 must have the same batch size."
            )

        if h_clin.size(0) != batch_size:
            raise ValueError(
                "C2, C3, and h_clin must have the same batch size."
            )

        if c2.size(1) != self.c2_channels:
            raise ValueError(
                f"Expected C2 to have {self.c2_channels} channels, "
                f"but received {c2.size(1)}."
            )

        if c3.size(1) != self.c3_channels:
            raise ValueError(
                f"Expected C3 to have {self.c3_channels} channels, "
                f"but received {c3.size(1)}."
            )

        if h_clin.size(1) != self.clinical_dim:
            raise ValueError(
                f"Expected h_clin dimension {self.clinical_dim}, "
                f"but received {h_clin.size(1)}."
            )

        # ========================================================
        # A. Global Hypothesis-Guided Verification
        # ========================================================

        # --------------------------------------------------------
        # T_1 = Flatten(C_2)
        # T_2 = Flatten(C_3)
        # --------------------------------------------------------

        t1 = self._flatten_feature_map(c2)
        t2 = self._flatten_feature_map(c3)

        # --------------------------------------------------------
        # phi_1(h_clin)
        # phi_2(h_clin)
        #
        # Add a query-token dimension:
        # [B, D] -> [B, 1, D]
        # --------------------------------------------------------

        q1 = self.phi_1(h_clin).unsqueeze(1)
        q2 = self.phi_2(h_clin).unsqueeze(1)

        # --------------------------------------------------------
        # o_1 = MHA(phi_1(h_clin), T_1, T_1)
        # --------------------------------------------------------

        o1, attn_c2 = self.global_attention_c2(
            query=q1,
            key=t1,
            value=t1,
            need_weights=True,
            average_attn_weights=True,
        )

        # --------------------------------------------------------
        # o_2 = MHA(phi_2(h_clin), T_2, T_2)
        # --------------------------------------------------------

        o2, attn_c3 = self.global_attention_c3(
            query=q2,
            key=t2,
            value=t2,
            need_weights=True,
            average_attn_weights=True,
        )

        # [B, 1, D] -> [B, D]
        o1 = o1.squeeze(1)
        o2 = o2.squeeze(1)

        # --------------------------------------------------------
        # h_global = psi([o_1; o_2])
        # --------------------------------------------------------

        global_concat = torch.cat(
            [o1, o2],
            dim=-1,
        )

        h_global = self.psi(
            global_concat
        )

        # ========================================================
        # B. Local Saliency-Guided Verification
        # ========================================================

        # --------------------------------------------------------
        # S = Conv_1x1(C3)
        # --------------------------------------------------------

        saliency_map = self.saliency_head(
            c3
        )

        # --------------------------------------------------------
        # {P_i}_{i=1}^{K} = SampleTopK(C2, S)
        # --------------------------------------------------------

        patches, topk_scores, topk_indices = self.patch_sampler(
            feature_map=c2,
            saliency_map=saliency_map,
        )

        # patches:
        # [B, K, C2, P, P]

        # --------------------------------------------------------
        # Flatten(P_i)
        # --------------------------------------------------------

        flattened_patches = patches.flatten(
            start_dim=2
        )

        # --------------------------------------------------------
        # r_i = rho(Flatten(P_i))
        #
        # [B, K, C2*P*P] -> [B, K, local_dim]
        # --------------------------------------------------------

        local_tokens = self.rho(
            flattened_patches
        )

        # --------------------------------------------------------
        # Construct:
        #
        # [r_agg; r_1; ...; r_K]
        # --------------------------------------------------------

        aggregate_token = self.r_agg.expand(
            batch_size,
            -1,
            -1,
        )

        local_sequence = torch.cat(
            [
                aggregate_token,
                local_tokens,
            ],
            dim=1,
        )

        # --------------------------------------------------------
        # TransformerEnc(
        #     [r_agg; r_1; ...; r_K]
        # )
        # --------------------------------------------------------

        encoded_local_sequence = self.local_transformer(
            local_sequence
        )

        # --------------------------------------------------------
        # h_local =
        # [
        # TransformerEnc(
        #   [r_agg; r_1; ...; r_K]
        # )
        # ]_0
        # --------------------------------------------------------

        h_local = encoded_local_sequence[:, 0, :]

        # ========================================================
        # C. Global-Local Visual Representation
        # ========================================================

        # --------------------------------------------------------
        # h_vis =
        # MLP_vis([h_global; h_local])
        # --------------------------------------------------------

        global_local_concat = torch.cat(
            [
                h_global,
                h_local,
            ],
            dim=-1,
        )

        h_vis = self.visual_mlp(
            global_local_concat
        )

        # ========================================================
        # D. Visual Prediction
        # ========================================================

        # --------------------------------------------------------
        # z_v = f_v(h_vis)
        # --------------------------------------------------------

        logits = self.visual_classifier(
            h_vis
        )

        # --------------------------------------------------------
        # p_vis = softmax(z_v)
        # --------------------------------------------------------

        p_vis = F.softmax(
            logits,
            dim=-1,
        )

        # --------------------------------------------------------
        # Visual evidential representation:
        #
        # evidence_v = softplus(z_v)
        #
        # alpha_v = evidence_v + 1
        #         = softplus(z_v) + 1
        # --------------------------------------------------------

        evidence = F.softplus(
            logits
        )

        alpha_vis = evidence + 1.0

        # ========================================================
        # Output
        # ========================================================

        outputs = {
            "h_global": h_global,
            "h_local": h_local,
            "h_vis": h_vis,
            "logits": logits,
            "p_vis": p_vis,
            "evidence": evidence,
            "alpha_vis": alpha_vis,
        }

        if return_auxiliary:
            outputs.update(
                {
                    "saliency_map": saliency_map,
                    "topk_scores": topk_scores,
                    "topk_indices": topk_indices,
                    "local_patches": patches,
                    "local_tokens": local_tokens,
                    "local_sequence": encoded_local_sequence,
                    "attention_c2": attn_c2,
                    "attention_c3": attn_c3,
                }
            )

        return outputs


# Short alias used by the main HVNet model.
MGVA = MultiGranularityVisualAttention

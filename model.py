"""
model.py

Fusion CNN + metadata model for SEM regularity prediction.

Architecture:
    image -> ResNet18 backbone -> img_feat (512)
    meta  -> small MLP         -> meta_feat (32)
    concat -> fused_feat       -> regression head + classification head

Outputs:
    - score_pred : continuous Gini prediction
    - logits     : class predictions (4 bins)
"""

import torch
import torch.nn as nn
from torchvision import models


class ArestyRegClsModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        use_pretrained=False,
        meta_dim=6,
        meta_feat_dim=32,
    ):
        super().__init__()

        # -------------------------
        # Backbone: ResNet18
        # -------------------------
        # Use pretrained ImageNet weights if enabled
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        backbone = models.resnet18(weights=weights)

        # -------------------------
        # Modify first convolution layer
        # -------------------------
        # Default ResNet expects 3-channel RGB input
        # We adapt it to grayscale (1 channel) or custom channel count
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # If using pretrained weights, convert RGB weights → grayscale
        # by averaging across input channels
        if use_pretrained and old_conv.weight is not None:
            with torch.no_grad():
                if in_channels == 1:
                    # Average RGB filters → single channel
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    # Extend to multiple channels if needed (future use)
                    w_gray = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 0:1].copy_(w_gray)
                    if in_channels > 1:
                        new_conv.weight[:, 1:2].copy_(w_gray)

        backbone.conv1 = new_conv

        # -------------------------
        # Feature extractor
        # -------------------------
        # Remove final FC layer → output shape: (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.img_feat_dim = 512

        # -------------------------
        # Metadata branch (MLP)
        # -------------------------
        # Input:
        # [fluence, delay, pulses, fluence_mask, delay_mask, pulses_mask]
        # Masks indicate whether metadata is present (handles missing values)
        self.meta_feat_dim = meta_feat_dim
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, meta_feat_dim),
            nn.ReLU(),
        )

        # -------------------------
        # Fusion
        # -------------------------
        # Concatenate image + metadata features
        fused_dim = self.img_feat_dim + self.meta_feat_dim

        # -------------------------
        # Regression head
        # -------------------------
        # Predict continuous Gini score
        self.reg_head = nn.Linear(fused_dim, 1)

        # -------------------------
        # Classification head
        # -------------------------
        # Predict 4 regularity classes
        # Includes hidden layer + dropout for better generalization
        self.cls_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, meta=None):
        """
        Forward pass.

        Args:
            x    : image tensor (B, 1, H, W)
            meta : metadata tensor (B, 6)

        Returns:
            score_pred : regression output (B,)
            logits     : classification output (B, num_classes)
        """

        # -------------------------
        # Image branch
        # -------------------------
        img_feat = self.backbone(x).flatten(1)  # shape: (B, 512)

        # -------------------------
        # Metadata branch
        # -------------------------
        if meta is None:
            # If metadata not provided, use zeros (graceful fallback)
            meta_feat = torch.zeros(
                (img_feat.size(0), self.meta_feat_dim),
                device=img_feat.device,
                dtype=img_feat.dtype,
            )
        else:
            meta_feat = self.meta_mlp(meta)

        # -------------------------
        # Feature fusion
        # -------------------------
        feat = torch.cat([img_feat, meta_feat], dim=1)

        # -------------------------
        # Outputs
        # -------------------------
        score_pred = self.reg_head(feat).squeeze(1)  # (B,)
        logits = self.cls_head(feat)

        return score_pred, logits
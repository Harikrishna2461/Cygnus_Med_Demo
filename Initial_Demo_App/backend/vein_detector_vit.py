"""
Custom Vision Transformer for Ultrasound Vein Detection
Designed for real-time detection of fascia and veins with N1/N2/N3 classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class VeinDetectionConfig:
    """Configuration for vein detection model"""
    image_size: int = 512
    patch_size: int = 16
    embedding_dim: int = 768
    num_blocks: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    num_classes: int = 4  # background, fascia, veins, uncertain
    vein_classes: int = 3  # N1, N2, N3


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, config: VeinDetectionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embedding_dim = config.embedding_dim

        # Linear projection of patches
        self.projection = nn.Conv2d(
            3, config.embedding_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Number of patches
        num_patches = (config.image_size // config.patch_size) ** 2

        # Learnable positional embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config.embedding_dim)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, height, width)
        Returns:
            embeddings: (batch_size, num_patches + 1, embedding_dim)
        """
        batch_size = x.size(0)

        # Project patches
        x = self.projection(x)  # (batch_size, embedding_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch_size, embedding_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embedding_dim)

        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (batch_size, num_patches + 1, embedding_dim)

        # Add positional embeddings
        x = x + self.position_embeddings

        return x


class SpatialAttention(nn.Module):
    """Multi-head spatial attention for understanding vein structures"""
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embedding_dim)
        Returns:
            output: (batch_size, seq_len, embedding_dim)
        """
        batch_size = x.size(0)

        Q = self.query(x)  # (batch_size, seq_len, embedding_dim)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.embedding_dim)

        output = self.fc_out(context)
        return output


class CrossAttention(nn.Module):
    """Cross-attention for fascia-aware vein detection"""
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Query from veins, Key/Value from fascia
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vein_features: torch.Tensor, fascia_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vein_features: (batch_size, seq_len, embedding_dim)
            fascia_features: (batch_size, seq_len, embedding_dim)
        Returns:
            output: (batch_size, seq_len, embedding_dim)
        """
        batch_size = vein_features.size(0)

        Q = self.query(vein_features)
        K = self.key(fascia_features)
        V = self.value(fascia_features)

        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.embedding_dim)

        output = self.fc_out(context)
        return output


class TransformerBlock(nn.Module):
    """Single transformer block with spatial attention"""
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()

        # Layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        # Spatial attention
        self.attention = SpatialAttention(embedding_dim, num_heads, dropout)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer block with residual connections"""
        # Attention with residual
        x = x + self.attention(self.ln1(x))

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x


class CustomUltrasoundViT(nn.Module):
    """Custom Vision Transformer for ultrasound vein detection"""
    def __init__(self, config: VeinDetectionConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)

        # Transformer blocks (for feature extraction)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embedding_dim, config.num_heads, config.mlp_dim, config.dropout)
            for _ in range(config.num_blocks)
        ])

        # Fascia detection head
        self.fascia_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, 2)  # fascia / no fascia
        )

        # Vein segmentation head (instance segmentation)
        self.vein_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.num_classes)  # background, vein classes
        )

        # Vein classification head (N1/N2/N3)
        self.classification_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.vein_classes)  # N1, N2, N3
        )

        # Cross-attention for fascia-aware vein understanding
        self.cross_attention = CrossAttention(config.embedding_dim, config.num_heads, config.dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, 3, height, width) - RGB ultrasound image
        Returns:
            dict with:
                - fascia_logits: (batch_size, num_patches + 1, 2)
                - vein_logits: (batch_size, num_patches + 1, num_classes)
                - classification_logits: (batch_size, num_patches + 1, vein_classes)
        """
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, num_patches + 1, embedding_dim)

        # Store fascia features for cross-attention
        fascia_features = x.clone()
        vein_features = x.clone()

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Update feature representations
        fascia_features = x
        vein_features = x

        # Cross-attention: enhance vein features with fascia context
        vein_features = self.cross_attention(vein_features, fascia_features)

        # Prediction heads
        fascia_logits = self.fascia_head(fascia_features)  # (batch_size, num_patches + 1, 2)
        vein_logits = self.vein_head(vein_features)  # (batch_size, num_patches + 1, num_classes)
        classification_logits = self.classification_head(vein_features)  # (batch_size, num_patches + 1, vein_classes)

        return {
            'fascia_logits': fascia_logits,
            'vein_logits': vein_logits,
            'classification_logits': classification_logits,
            'features': x
        }


class VeinDetectionPostProcessor:
    """Post-processing for vein detection outputs"""

    def __init__(self, image_size: int = 512, patch_size: int = 16):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

    def logits_to_segmentation(self, logits: torch.Tensor) -> np.ndarray:
        """Convert logits to segmentation map"""
        # Remove CLS token
        logits = logits[:, 1:, :]  # (batch_size, num_patches, num_classes)

        # Get class predictions
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, num_patches)

        # Reshape to spatial grid
        grid_size = int(np.sqrt(self.num_patches))
        segmentation = predictions.view(-1, grid_size, grid_size)  # (batch_size, grid_h, grid_w)

        # Upsample to original image size
        segmentation = segmentation.unsqueeze(1).float()  # (batch_size, 1, grid_h, grid_w)
        segmentation = F.interpolate(
            segmentation,
            size=(self.image_size, self.image_size),
            mode='nearest'
        )
        segmentation = segmentation.squeeze(1).cpu().numpy()

        return segmentation

    def classify_veins(self, logits: torch.Tensor) -> Dict[str, np.ndarray]:
        """Classify detected veins as N1, N2, or N3"""
        # Remove CLS token
        logits = logits[:, 1:, :]  # (batch_size, num_patches, vein_classes)

        # Get classifications
        classifications = torch.argmax(logits, dim=-1)  # (batch_size, num_patches)
        probabilities = F.softmax(logits, dim=-1)  # (batch_size, num_patches, vein_classes)

        # Map to N1, N2, N3
        class_names = ['N1', 'N2', 'N3']
        classification_names = torch.tensor([class_names[c] for c in classifications.cpu().numpy().flatten()])

        return {
            'classifications': classifications.cpu().numpy(),
            'class_names': classification_names.numpy(),
            'probabilities': probabilities.cpu().numpy()
        }


def create_model(config: VeinDetectionConfig = None) -> CustomUltrasoundViT:
    """Factory function to create model"""
    if config is None:
        config = VeinDetectionConfig()
    return CustomUltrasoundViT(config)


if __name__ == "__main__":
    # Test the model
    config = VeinDetectionConfig()
    model = create_model(config)

    # Create dummy input
    x = torch.randn(2, 3, 512, 512)

    # Forward pass
    output = model(x)

    print("Model output shapes:")
    print(f"  fascia_logits: {output['fascia_logits'].shape}")
    print(f"  vein_logits: {output['vein_logits'].shape}")
    print(f"  classification_logits: {output['classification_logits'].shape}")

    # Post-processing
    post_processor = VeinDetectionPostProcessor()
    segmentation = post_processor.logits_to_segmentation(output['vein_logits'])
    classifications = post_processor.classify_veins(output['classification_logits'])

    print(f"\nSegmentation output shape: {segmentation.shape}")
    print(f"Classification output shape: {classifications['classifications'].shape}")

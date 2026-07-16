import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 norm=None, activation=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deformable_groups=1, bias=False, norm=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias)
        self.norm = norm
        self.activation = activation

    def forward(self, x, offset):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm, out_channels):
    if norm is None or norm == '':
        return None
    if norm == 'BN':
        return nn.BatchNorm2d(out_channels)
    if norm == 'GN':
        return nn.GroupNorm(32, out_channels)
    if norm == 'LN':
        return nn.LayerNorm(out_channels)
    return None


def cat(tensors, dim=0):
    return torch.cat(tensors, dim=dim)


def shapes_to_tensor(x, device=None):
    if torch.jit.is_scripting():
        return torch.as_tensor(x, dtype=torch.int32, device=device)
    if device is None:
        return torch.as_tensor(x, dtype=torch.int32)
    return torch.as_tensor(x, dtype=torch.int32, device=device)

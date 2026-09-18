"""
Attention U-Net for binary vein segmentation.

- Standard U-Net encoder/decoder backbone.
- Attention gates on every skip connection (Oktay et al., "Attention U-Net"),
  so the decoder learns to suppress irrelevant background regions of the
  skip features using the coarser gating signal from below.
- A CBAM (channel + spatial attention) block applied to the final decoder
  feature map, right before the 1x1 classification-head convolution, so the
  head itself gets an extra attention-refined feature map to classify from.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """Additive attention gate: gates skip-connection features `x` using the
    coarser decoder signal `g`."""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = torch.mean(x, dim=(2, 3))
        mx, _ = torch.max(x.view(b, c, -1), dim=2)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Channel + spatial attention block used right before the segmentation
    classification head."""

    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.bottleneck = conv_block(c4, c5)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.att4 = AttentionGate(c4, c4, c4 // 2)
        self.dec4 = conv_block(c5, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.att3 = AttentionGate(c3, c3, c3 // 2)
        self.dec3 = conv_block(c4, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.att2 = AttentionGate(c2, c2, c2 // 2)
        self.dec2 = conv_block(c3, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.att1 = AttentionGate(c1, c1, c1 // 2)
        self.dec1 = conv_block(c2, c1)

        # attention-refined classification head
        self.head_cbam = CBAM(c1)
        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4a = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4a], dim=1))

        d3 = self.up3(d4)
        e3a = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))

        d2 = self.up2(d3)
        e2a = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))

        d1 = self.up1(d2)
        e1a = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))

        feat = self.head_cbam(d1)
        logits = self.head(feat)
        return logits


if __name__ == "__main__":
    m = AttentionUNet()
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print(y.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params/1e6:.2f}M")

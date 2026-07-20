"""
Enhanced UNet with attention mechanisms and architectural awareness
for better wall/window/door detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Attention Modules
# -------------------------
class AttentionBlock(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

# -------------------------
# Enhanced Blocks
# -------------------------
def conv_block_enhanced(in_c, out_c, kernel=4, stride=2, padding=1, use_bn=True, use_attention=False):
    layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    
    block = nn.Sequential(*layers)
    
    if use_attention:
        return nn.Sequential(
            block,
            ChannelAttention(out_c),
            SpatialAttention()
        )
    return block

def deconv_block_enhanced(in_c, out_c, kernel=4, stride=2, padding=1, use_dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

# -------------------------
# Enhanced UNet Generator
# -------------------------
class EnhancedUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, ngf=64):
        super().__init__()
        
        # Encoder with attention at deeper layers
        self.enc1 = conv_block_enhanced(in_channels, ngf, use_bn=False)  # 128x128
        self.enc2 = conv_block_enhanced(ngf, ngf*2)  # 64x64
        self.enc3 = conv_block_enhanced(ngf*2, ngf*4, use_attention=True)  # 32x32
        self.enc4 = conv_block_enhanced(ngf*4, ngf*8, use_attention=True)  # 16x16
        self.enc5 = conv_block_enhanced(ngf*8, ngf*8, use_attention=True)  # 8x8
        self.enc6 = conv_block_enhanced(ngf*8, ngf*8)  # 4x4
        self.enc7 = conv_block_enhanced(ngf*8, ngf*8)  # 2x2
        
        # Bottleneck with residual connection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1),
            nn.ReLU(True)
        )
        
        # Attention gates for skip connections
        self.att7 = AttentionBlock(F_g=ngf*8, F_l=ngf*8, F_int=ngf*4)
        self.att6 = AttentionBlock(F_g=ngf*8, F_l=ngf*8, F_int=ngf*4)
        self.att5 = AttentionBlock(F_g=ngf*8, F_l=ngf*8, F_int=ngf*4)
        self.att4 = AttentionBlock(F_g=ngf*8, F_l=ngf*8, F_int=ngf*4)
        self.att3 = AttentionBlock(F_g=ngf*4, F_l=ngf*4, F_int=ngf*2)
        self.att2 = AttentionBlock(F_g=ngf*2, F_l=ngf*2, F_int=ngf)
        self.att1 = AttentionBlock(F_g=ngf, F_l=ngf, F_int=ngf//2)
        
        # Decoder with attention-weighted skip connections
        self.dec1 = deconv_block_enhanced(ngf*8, ngf*8, use_dropout=True)  # 1x1 -> 2x2
        self.dec2 = deconv_block_enhanced(ngf*8*2, ngf*8, use_dropout=True)  # 2x2 -> 4x4
        self.dec3 = deconv_block_enhanced(ngf*8*2, ngf*8, use_dropout=True)  # 4x4 -> 8x8
        self.dec4 = deconv_block_enhanced(ngf*8*2, ngf*8)  # 8x8 -> 16x16
        self.dec5 = deconv_block_enhanced(ngf*8*2, ngf*4)  # 16x16 -> 32x32
        self.dec6 = deconv_block_enhanced(ngf*4*2, ngf*2)  # 32x32 -> 64x64
        self.dec7 = deconv_block_enhanced(ngf*2*2, ngf)  # 64x64 -> 128x128
        
        # Final layer to get to output size (128x128 -> 256x256)
        # FIXED: Input channels changed from ngf to ngf*2 to match d7_cat
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_channels, 4, 2, 1),  # ← Changed from ngf to ngf*2
            nn.Tanh()
        )
        
        # Edge detection branch (takes input after concatenation with skip)
        self.edge_branch = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, 1, 1),  # Takes ngf*2 channels
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf//2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(ngf//2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        # Bottleneck
        b = self.bottleneck(e7)
        
        # Decoder with attention gates
        d1 = self.dec1(b)
        e7_att = self.att7(d1, e7)
        d1 = torch.cat([d1, e7_att], dim=1)
        
        d2 = self.dec2(d1)
        e6_att = self.att6(d2, e6)
        d2 = torch.cat([d2, e6_att], dim=1)
        
        d3 = self.dec3(d2)
        e5_att = self.att5(d3, e5)
        d3 = torch.cat([d3, e5_att], dim=1)
        
        d4 = self.dec4(d3)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        
        d5 = self.dec5(d4)
        e3_att = self.att3(d5, e3)
        d5 = torch.cat([d5, e3_att], dim=1)
        
        d6 = self.dec6(d5)
        e2_att = self.att2(d6, e2)
        d6 = torch.cat([d6, e2_att], dim=1)
        
        d7 = self.dec7(d6)
        e1_att = self.att1(d7, e1)
        d7_cat = torch.cat([d7, e1_att], dim=1)  # Now has ngf*2 = 128 channels
        
        # Main output - uses d7_cat (128 channels)
        out = self.final(d7_cat)
        
        # Edge output - uses d7_cat (128 channels)
        edge = self.edge_branch(d7_cat)
        
        return out, edge

# -------------------------
# Keep original discriminator
# -------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, ndf=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, 1, 1)
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, input_rgb, target):
        x = torch.cat([input_rgb, target], dim=1)
        return self.model(x)
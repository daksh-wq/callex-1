import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetResidualBlock(nn.Module):
    """
    Tier-1 Generative Dilated Convolutions native to WaveNet architectures.
    Forces the Generative Math to look backwards at previous acoustic samples (Dilation) 
    without destroying the explicit forward sequence.
    """
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_res = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, mask=None):
        # Neural gating separates harmonic mathematical logic from pure noise variables
        x_filter = torch.tanh(self.conv_filter(x))
        x_gate = torch.sigmoid(self.conv_gate(x))
        
        # Element-wise acoustic merging
        z = x_filter * x_gate
        
        if mask is not None:
             z = z * mask
             
        res = self.conv_res(z)
        skip = self.conv_skip(z)
        return (x + res) * (0.5 ** 0.5), skip

class AffineCouplingLayer(nn.Module):
    """
    Continuous Normalizing Flow physics executing bijective transformations natively.
    Reverses the explicit probabilistic boundaries of Audio Mel-Spectrograms so the Model 
    can legally generate speech from pure Gaussian White Noise!
    """
    def __init__(self, channels=192, hidden_channels=192, depth=4):
        super().__init__()
        self.pre_conv = nn.Conv1d(channels // 2, hidden_channels, 1)
        
        # Dilated logic stacks ensuring deep temporal analysis natively
        self.wavenet_blocks = nn.ModuleList([
            WaveNetResidualBlock(hidden_channels, 2 ** i) for i in range(depth)
        ])
        self.post_conv = nn.Conv1d(hidden_channels, channels, 1)
        
        # Initialize final mapping gracefully effectively at zero to ensure stable execution dynamically
        nn.init.zeros_(self.post_conv.weight)
        nn.init.zeros_(self.post_conv.bias)

    def forward(self, x, mask=None, reverse=False):
        # Coupling mechanically splits spatial flow mapping tensors strictly natively
        x0, x1 = torch.split(x, x.size(1) // 2, dim=1)
        
        h = self.pre_conv(x0)
        skip_accum = 0
        
        for block in self.wavenet_blocks:
            h, skip = block(h, mask)
            skip_accum += skip
            
        h = skip_accum * (1 / len(self.wavenet_blocks) ** 0.5)
        m, logs = torch.split(self.post_conv(h), x.size(1) // 2, dim=1)
        
        if not reverse:
            x1 = x1 * torch.exp(logs) + m
            z = torch.cat([x0, x1], dim=1)
            return z, logs
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            z = torch.cat([x0, x1], dim=1)
            return z

class ResidualPosteriorEncoder(nn.Module):
    """
    Replaces basic convolutions. A deep Normalizing Matrix transforming Human Waveforms 
    into absolute normal prior distributions natively measuring mathematically pure KL parameters!
    """
    def __init__(self, in_channels=80, hidden_channels=192, out_channels=192):
        super().__init__()
        self.prequeeze = nn.Conv1d(in_channels, hidden_channels, 1)
        self.wavenet = nn.ModuleList([
            WaveNetResidualBlock(hidden_channels, 2 ** (i % 4)) for i in range(8)
        ])
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x):
        x = self.prequeeze(x)
        for block in self.wavenet:
            x, _ = block(x)
        stats = self.proj(x)
        mu, logs = torch.split(stats, stats.size(1) // 2, 1)
        
        # Execute explicitly mathematically sound reparameterization trick dynamically
        z = mu + torch.randn_like(mu) * torch.exp(logs)
        return z, mu, logs

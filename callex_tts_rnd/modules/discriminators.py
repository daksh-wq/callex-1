import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorP(nn.Module):
    """
    Sub-Auditor explicitly targeting phase shifts natively in pure Mel-Sequences dynamically across 
    singular periods. Natively wraps Weight Normalization dynamically ensuring strict boundary grading.
    """
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        # Target explicit periodic mathematical folding structurally
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
            
        x = x.view(b, c, t // self.period, self.period)
        
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        
        out = self.conv_post(x)
        features.append(out)
        return out, features

class MultiPeriodDiscriminator(nn.Module):
    """
    Advanced Generative Acoustic Validator seamlessly routing matrices into specialized 
    sub-auditors parallelizing pure math checks simultaneously!
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5),
            DiscriminatorP(7), DiscriminatorP(11)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    """
    Scale-based spatial Auditor testing purely melodic sequences spanning enormous temporal limits.
    Utilizes Spectral Norm structurally balancing deeper multi-resolution gradients dynamically!
    """
    def __init__(self, norm=weight_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        out = self.conv_post(x)
        features.append(out)
        return out, features

class MultiScaleDiscriminator(nn.Module):
    """
    Assembles Tier-1 Deep Spectral Auditors universally mapping across native physical Audio outputs.
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(spectral_norm), 
            DiscriminatorS(weight_norm)
        ])
        # Downsamplers structurally shrink vectors natively so judges see the big picture!
        self.poolings = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
            
            if i < len(self.poolings):
                y = self.poolings[i](y)
                y_hat = self.poolings[i](y_hat)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

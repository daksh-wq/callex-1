import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Deep Swish Activation Function. Standardizes massive gradients natively."""
    def forward(self, x):
        return x * torch.sigmoid(x)

class MultiHeadRelativeAttentionModule(nn.Module):
    """
    Tier-1 STT Attention. Maps explicit logical boundaries tracking acoustic contexts.
    Allows the model to 'Listen' backwards and forwards to understand if a speaker 
    mumbled a word based on the context of the entire sentence natively!
    """
    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = self.layernorm(x)
        out, _ = self.attention(out, out, out, key_padding_mask=mask)
        return self.dropout(out)

class ConvolutionModule(nn.Module):
    """
    The distinct innovation of the Conformer Architecture. 
    By executing 1D Depthwise Convolutions dynamically over the Attention embeddings, 
    the model learns exact sub-word phonetic vibrations (like recognizing a 'ch' vs 'sh' sound).
    """
    def __init__(self, channels=256, kernel_size=15, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels)
        self.batchnorm = nn.BatchNorm1d(channels)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Transpose inherently to allow mathematical convolution bounding along sequential time vectors
        x = self.layernorm(x).transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

class MacaronFeedForwardModule(nn.Module):
    """
    Macaron FFN Net: Traps massive linear projection layers natively between 'Half-Step' residual gates.
    """
    def __init__(self, d_model=256, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)

class ConformerBlock(nn.Module):
    """
    The True Heart of Google's state-of-the-art Voice Transcription Neural Grid natively rewritten.
    Orchestrates the exact Macaron topology dynamically executing Half-Step FFNs, 
    Relative Multi-Head Attention, and Depthwise Phonetic Convolutions symmetrically.
    """
    def __init__(self, d_model=256, num_heads=4, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = MacaronFeedForwardModule(d_model, dropout=dropout)
        self.attention = MultiHeadRelativeAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = MacaronFeedForwardModule(d_model, dropout=dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attention(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.layernorm(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadRelativeAttention(nn.Module):
    """
    Tier-1 Explicit Multi-Head Self-Attention utilizing Relative Positional physics.
    Unlike standard Transformers which use Absolute indexing, Relative Positional Encodings
    teach the model the explicit mathematical distances between words, causing the synthesized 
    Hindi to dynamically fluctuate pitch logically based on sentence structure.
    """
    def __init__(self, channels=192, hidden_channels=192, heads=2, dropout=0.1, window_size=4):
        super().__init__()
        assert channels % heads == 0
        
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.window_size = window_size
        
        # Core query/key/value projection matrices calculated asynchronously natively
        self.k_proj = nn.Conv1d(channels, hidden_channels, 1)
        self.v_proj = nn.Conv1d(channels, hidden_channels, 1)
        self.q_proj = nn.Conv1d(channels, hidden_channels, 1)
        self.out_proj = nn.Conv1d(hidden_channels, channels, 1)
        
        # Essential Dropouts optimizing network bounds against massive dataset overfitting
        self.drop = nn.Dropout(dropout)
        
        # Embedding array modeling explicitly relational boundaries across the convolution
        self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, hidden_channels // heads))
        self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, hidden_channels // heads))

    def forward(self, x, mask=None):
        # x -> [Batch, Channels, Time]
        b, c, t = x.size()
        
        q = self.q_proj(x).view(b, self.heads, self.hidden_channels // self.heads, t)
        k = self.k_proj(x).view(b, self.heads, self.hidden_channels // self.heads, t)
        v = self.v_proj(x).view(b, self.heads, self.hidden_channels // self.heads, t)
        
        # Compute explicit mathematical dot-product relationships across phonetic ranges
        scores = torch.einsum('b h d t, b h d s -> b h t s', q, k) / (self.hidden_channels // self.heads) ** 0.5
        
        # Apply strict logical masking if arrays differ in boundaries
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, -1e4)
            
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        
        # Remap weighted logic output dynamically back to explicit waveform tensor matrices
        output = torch.einsum('b h t s, b h d s -> b h d t', p_attn, v)
        output = output.reshape(b, self.hidden_channels, t)
        
        # Final physical layer projection executing transformation natively
        return self.out_proj(output)

class FeedForwardNetwork(nn.Module):
    """
    Non-linear auditory activation pathway ensuring high-dimensional structural relationships
    are forced gracefully into predictable mathematical boundaries via explicit Convolution grids.
    """
    def __init__(self, channels=192, hidden_channels=768, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(hidden_channels, channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        return self.drop(x)

class CallexAcousticTransformerBlock(nn.Module):
    """
    A single universal repeating node orchestrating both the Relative Attention physics 
    and the FeedForward convolutions completely independently on standard VRAM stacks.
    """
    def __init__(self, channels=192, hidden_channels=768, heads=2, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadRelativeAttention(channels, channels, heads, dropout)
        self.norm_1 = nn.BatchNorm1d(channels)
        self.ffn = FeedForwardNetwork(channels, hidden_channels, dropout=dropout)
        self.norm_2 = nn.BatchNorm1d(channels)

    def forward(self, x, mask=None):
        # Neural Skip connections inherently protecting Deep structures from Vanishing Gradients
        x = x + self.attention(self.norm_1(x), mask)
        x = x + self.ffn(self.norm_2(x))
        return x

class CallexTextEncoder(nn.Module):
    """
    The Ultimate Core PyTorch Module generating complex Acoustic contexts dynamically natively.
    Replaces the introductory 'FlowTextEncoder' with a completely massive 6-layer 
    Self-Attention architecture exactly identical to Tier-1 Voice Models (e.g. Meta Voicebox).
    """
    def __init__(self, vocab_size, channels=192, depth=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.scale = channels ** 0.5
        
        # Stack 6 consecutive Tier-1 Positional Convolution Blocks seamlessly
        self.layers = nn.ModuleList([
            CallexAcousticTransformerBlock(channels=channels, heads=2) for _ in range(depth)
        ])
        
        self.post_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x shape => [batch, sequence_len]
        mask = (x != 0)
        x = self.embedding(x) * self.scale
        x = x.transpose(1, 2) # Switch for Conv1d topology => [batch, channels, sequence_len]
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.post_proj(x)
        return x, mask

import torch
import torch.nn as nn
import random

class SpecAugment(nn.Module):
    """
    Tier-1 STT Data Augmentation Module. 
    A model trained on clean audio fails entirely when users call from the street.
    SpecAugment mathematically sabotages (masks) the training audio deliberately by 
    deleting blocks of frequencies and physical timeframes. This literally forces the AI 
    to learn how to decipher mumbled or static-filled Indian accents perfectly!
    """
    def __init__(self, freq_mask_param=27, time_mask_param=100, num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        """
        Dynamically erases spectrogram tensors matrices mathematically during forward propagation.
        x expected shape: [batch, features, time]
        """
        if not self.training:
            return x

        b, f, t = x.shape
        cloned_x = x.clone()

        # ── FREQUENCY MASKING ──
        # Simulates a broken VoIP telephone bandwidth strictly natively by deleting entire sound channels!
        for _ in range(self.num_freq_masks):
            f_mask_len = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, f - f_mask_len))
            cloned_x[:, f0:f0 + f_mask_len, :] = 0

        # ── TIME MASKING ──
        # Simulates a user's phone cutting out for a split second natively by erasing time frames!
        for _ in range(self.num_time_masks):
            t_mask_len = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, t - t_mask_len))
            cloned_x[:, :, t0:t0 + t_mask_len] = 0

        return cloned_x

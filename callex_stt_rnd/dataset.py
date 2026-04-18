import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import os

class SpectralAugment(torch.nn.Module):
    """
    SpecAugment randomly masks frequency bins and time blocks of the MFCC feature 
    map to force the neural network to handle noisy or interrupted audio inputs.
    """
    def __init__(self, freq_mask_param=30, time_mask_param=100):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        
    def forward(self, mel_spectrogram):
        # Apply sequential masking
        augmented = self.freq_mask(mel_spectrogram)
        augmented = self.time_mask(augmented)
        return augmented

class STTDataset(Dataset):
    """
    Dataset loader for Callex Native audio sets. Converts PCM audio pairs
    into MFCC tensors directly via Torchaudio, avoiding heavy preprocessing steps.
    """
    def __init__(self, manifest_file, tokenizer, is_training=True):
        self.data_rows = []
        # Typically parse mapping file (wav_path|transcript)
        # For boilerplate, we simulate an empty loop
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) == 2:
                        self.data_rows.append((parts[0], parts[1]))
        
        self.tokenizer = tokenizer
        self.is_training = is_training
        
        # 16kHz, 40 Coeff, 80 Mel bins as defined in R&D Doc
        self.mfcc_transform = T.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            log_mels=True,
            melkwargs={'n_fft': 512, 'n_mels': 80, 'hop_length': 160}
        )
        self.spec_augment = SpectralAugment(freq_mask_param=30, time_mask_param=100)
        
    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        audio_path, transcript = self.data_rows[idx]
        
        # Load audio tensor [1, T]
        waveform, sr = torchaudio.load(audio_path)
        
        # Process to MFCC features [1, 40, Frames]
        features = self.mfcc_transform(waveform)
        
        # Augment conditionally
        if self.is_training:
            features = self.spec_augment(features)
            
        # Add delta coefficients (+ delta + delta-delta = 120 features)
        # Note: Delta transforms are computational; simplified for base script
        features = features.squeeze(0).transpose(0, 1) # [Frames, 40]
        
        # Encode Target Transcript
        target_ids = self.tokenizer.encode(transcript)
        
        return features, torch.tensor(target_ids, dtype=torch.long)

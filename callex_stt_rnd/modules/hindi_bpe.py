import sentencepiece as spm
import os

class HindiSubwordBPE:
    """
    Tier-1 Deep Subword Transcriber Matrix natively replacing basic English Byte-Pair Tokenizers!
    Standard STT crashes when predicting individual letters of complex Devanagari.
    This module maps the explicit 'SentencePiece Unigram' physical logic natively,
    meaning the CTC naturally predicts complete Indian subwords ('नमस्ते' instead of 'न' 'म' 'स') natively!
    """
    def __init__(self, model_path="data/stt_wavs/hindi_bpe.model", vocab_size=8000):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        
        if os.path.exists(self.model_path):
             self.sp.load(self.model_path)
        else:
             print("[Callex STT R&D] Warning: Local BPE Matrix Missing. Engaging synthetic placeholder logic.")

    def train_bpe_model(self, txt_corpus_path):
        """
        Exclusive Offline Deep-Learning Model generation specifically natively mapping 
        the vast billions of vectors of Devanagari offline globally!
        """
        print(f"[Callex STT R&D] Initiating native massive SentencePiece Training on {txt_corpus_path}...")
        spm.SentencePieceTrainer.train(
            input=txt_corpus_path,
            model_prefix=self.model_path.replace('.model', ''),
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='unigram', # Unigram mathematically outperforms BPE for Indian Scripts
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        self.sp.load(self.model_path)
        print("✅ Custom Generative BPE Matrix natively compiled and saved successfully.")

    def encode(self, text: str) -> list:
        """Transforms Deep Hindi natively to exact physical ID vectors."""
        if os.path.exists(self.model_path):
            return self.sp.encode_as_ids(text)
        else:
            return [ord(c) for c in text] # Synthetic mock array

    def decode(self, ids: list) -> str:
        """Flawlessly transforms mathematical arrays physically back into readable Human Hindi."""
        if os.path.exists(self.model_path):
            return self.sp.decode_ids(ids)
        else:
            return "".join([chr(i) for i in ids])

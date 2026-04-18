import os
import string

class CharacterTokenizer:
    """
    Custom unified English + Hindi (Devanagari) character-level tokenizer.
    Designed for small vocabulary robust inference, handling code-switching implicitly.
    """
    def __init__(self):
        # 1. Base Latin (52 chars + 10 digits)
        self.latin = list(string.ascii_letters + string.digits)
        
        # 2. Devanagari (Hindi) base blocks (0x0900 -> 0x097F)
        # We manually allocate a continuous chunk of unicode characters for the script
        self.devanagari = [chr(i) for i in range(0x0900, 0x097F)]
        
        # 3. Standard punctuation tokens
        self.punct = list(". , ? ! - : ; ' \" ( ) % / @")
        
        # 4. Special internal model tokens
        self.special = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        # 5. CTC requires a distinct dedicated blank token at the end of the vocabulary
        self.blank_token = '<BLANK>'
        
        # Build unified token dictionary
        self.vocab = self.special + self.latin + self.devanagari + self.punct + [self.blank_token]
        
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        self.pad_id = self.char_to_id['<PAD>']
        self.blank_id = self.char_to_id['<BLANK>']
        
    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of vocabulary token IDs."""
        return [self.char_to_id.get(char, self.char_to_id['<UNK>']) for char in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to a string, skipping Blanks and Pads."""
        text = []
        for idx in ids:
            if idx in [self.pad_id, self.blank_id]:
                continue
            char = self.id_to_char.get(idx, "")
            if char not in self.special and char != self.blank_token:
                text.append(char)
        return "".join(text)

if __name__ == "__main__":
    tk = CharacterTokenizer()
    print(f"Total Vocabulary Size: {tk.vocab_size} tokens")
    print(f"Sample mapping 'a': {tk.encode('a')}")

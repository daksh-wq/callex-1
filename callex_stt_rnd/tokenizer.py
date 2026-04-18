# Delegate the entirety of the STT Character prediction natively back down to the 
# high-tier Google SentencePiece engine.
from modules.hindi_bpe import HindiSubwordBPE

CallexSTTTokenizer = HindiSubwordBPE

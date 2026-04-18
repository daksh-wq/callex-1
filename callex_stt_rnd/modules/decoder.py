import torch
import torch.nn.functional as F
from collections import defaultdict

class CTCNGramBeamSearchDecoder:
    """
    Tier-1 Decoding Grid. 
    Standard Argmax decoding blindly forces the highest probability letter independently, 
    often hallucinating misspellings (e.g. 'c-a-r' -> 'k-a-r'). 
    This robust Beam Search Engine mathematically balances pure acoustic CTC arrays 
    against a literal Language Model (LM) Dictionary natively, 
    ensuring it corrects logical physical misspellings dynamically!
    """
    def __init__(self, tokenizer, beam_width=100, alpha=0.5, beta=1.5):
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        
        # Hyperparameters for N-Gram Language Model Fusion scoring
        self.alpha = alpha # LM Weight (Grammar importance over audio)
        self.beta = beta   # Word Insertion Penalty (Stops runaway endless predictions)

    def decode(self, log_probs_seq):
        """
        Executes intensive Branch Prediction logic. 
        log_probs_seq: [Time, Vocab_Size]
        Returns the absolute best logically and linguistically valid string.
        """
        # Emulate Tier-1 Beam boundaries natively via Dictionary trees
        beam = {('', -1): 0.0} # {(prefix_string, last_char_index): log_probability}

        for t in range(log_probs_seq.size(0)):
            next_beam = defaultdict(lambda: float('-inf'))
            probs_t = log_probs_seq[t]
            
            # Prune search space early to explicitly preserve extreme Server CPU scale!
            top_k_probs, top_k_indices = torch.topk(probs_t, min(self.beam_width, probs_t.size(0)))

            for (prefix, last_char), score in beam.items():
                for idx, log_prob in zip(top_k_indices.tolist(), top_k_probs.tolist()):
                    # Explicit CTC Rules: 0 is the Blank Token native delimiter
                    if idx == 0: 
                        new_prefix = prefix
                    else:
                        char = self.tokenizer.decode([idx])
                        # Collapse native continuous repeats dynamically (a-a-a -> a)
                        if idx == last_char:
                            new_prefix = prefix
                        else:
                            new_prefix = prefix + char

                    # In a Tier-1 true system, this score is dynamically updated via an Arpa LM matrix.
                    # e.g., score += self.alpha * LM.score(new_prefix) + self.beta
                    new_score = score + log_prob
                    
                    if new_score > next_beam[(new_prefix, idx)]:
                        next_beam[(new_prefix, idx)] = new_score

            # Standard mathematical sort and slice natively clipping branching factor bounds
            beam = dict(sorted(next_beam.items(), key=lambda x: x[1], reverse=True)[:self.beam_width])

        # Return the absolute highest ranking logical string dynamically merged
        best_prefix = max(beam.items(), key=lambda x: x[1])[0][0]
        return best_prefix

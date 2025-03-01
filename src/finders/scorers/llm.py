import numpy as np
import torch

from src.finders.scorers.base import Scorer


class TransformerScorer(Scorer):
    """Transformer scorer to use log odds"""

    def __init__(self, model, tokenizer, top_n: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.top_n = top_n

    def score_candidates(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool | None = None,
    ) -> np.ndarray:
        if not candidates:
            return np.array([])

        # Prepare context
        context_str = " ".join(context)
        context_tokens = self.tokenizer(context_str, return_tensors="pt").input_ids

        with torch.no_grad():
            # Get next token logits (not probabilities!)
            outputs = self.model(context_tokens)
            next_token_logits = outputs.logits[0, -1, :]

            # Score each candidate using raw logits
            scores = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                candidate_tokens = self.tokenizer(candidate)["input_ids"]
                if candidate_tokens:
                    first_token_id = candidate_tokens[0]
                    # Use raw logit as log-odds score
                    scores[i] = next_token_logits[first_token_id].item()
                else:
                    scores[i] = float("-inf")

        return scores


class T5TransformerScorer(Scorer):
    """Scorer using T5 model for bidirectional context scoring"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def score_candidates(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool | None = None,
    ) -> np.ndarray:
        if not candidates:
            return np.array([])

        # Prepare context
        context_str = " ".join(context)
        scores = np.zeros(len(candidates))

        with torch.no_grad():
            for i, candidate in enumerate(candidates):
                # For T5, we'll prompt it to score the sequence
                if adding_right:
                    prompt = f"score sequence: {context_str} {candidate}"
                    target = "natural"  # Target for natural continuation
                else:
                    prompt = f"score sequence: {candidate} {context_str}"
                    target = "natural"  # Same target for consistency

                # Tokenize input and target
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                target_ids = self.tokenizer(target, return_tensors="pt").input_ids

                # Get model output
                outputs = self.model(**inputs, labels=target_ids)

                # Use loss as our score (negative because lower loss = better)
                scores[i] = -outputs.loss.item()  # Convert to positive score

                if scores[i] < -100:  # Clip extremely negative scores
                    scores[i] = float("-inf")

        return scores

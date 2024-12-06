import math
from collections import Counter
from dataclasses import dataclass

import nltk
import numpy as np
from nltk.tag import pos_tag

from src.finders.scorers.base import Scorer

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


@dataclass
class PartOfSpeechBalanceScorer(Scorer):
    """Rewards natural distribution of parts of speech"""

    # Ideal ratios based on typical English sentences
    # These are rough approximations - you may want to tune them
    target_ratios = {
        "NOUN": 0.25,  # Nouns (including proper nouns)
        "VERB": 0.20,  # Verbs (all forms)
        "ADJ": 0.15,  # Adjectives
        "ADP": 0.15,  # Prepositions/Adpositions
        "DET": 0.10,  # Determiners/Articles
        "ADV": 0.10,  # Adverbs
        "CCONJ": 0.05,  # Coordinating conjunctions
    }

    def get_weight(self, context_length: int) -> float:
        return min(1.0, context_length / 5)  # Ramps up to full weight at length 5

    def score_candidates(
        self, context: list[str], candidates: list[str], adding_right: bool | None = None
    ) -> np.ndarray:
        if not candidates:
            return np.array([])

        scores = np.zeros(len(candidates))

        # Get current POS distribution if we have context
        current_pos_counts = Counter()
        if context:
            tagged = pos_tag(context)
            for _, pos in tagged:
                # Map Penn Treebank tags to Universal tags
                pos = self._simplify_tag(pos)
                if pos in self.target_ratios:
                    current_pos_counts[pos] += 1

        # Score each candidate
        for i, candidate in enumerate(candidates):
            # Get candidate's POS
            tagged = pos_tag([candidate])[0][1]
            pos = self._simplify_tag(tagged)

            if pos not in self.target_ratios:
                continue

            # Calculate what ratio this POS would have if we added this word
            total_words = len(context) + 1
            new_ratio = (current_pos_counts[pos] + 1) / total_words
            target = self.target_ratios[pos]

            # Score based on how close we are to target ratio
            ratio_diff = abs(new_ratio - target)
            if ratio_diff > target * 2:  # Way too many of this POS
                scores[i] = float("-inf")
            else:
                # Logarithmic penalty for deviation from target
                scores[i] = -math.log1p(ratio_diff)

            # Scale score based on context length
            scores[i] *= self.get_weight(len(context))

        return scores

    def _simplify_tag(self, penn_tag: str) -> str:
        """Convert Penn Treebank tags to simplified Universal tags"""
        # Mapping of Penn Treebank tags to Universal tags
        tag_map = {
            "NN": "NOUN",
            "NNS": "NOUN",
            "NNP": "NOUN",
            "NNPS": "NOUN",
            "VB": "VERB",
            "VBD": "VERB",
            "VBG": "VERB",
            "VBN": "VERB",
            "VBP": "VERB",
            "VBZ": "VERB",
            "JJ": "ADJ",
            "JJR": "ADJ",
            "JJS": "ADJ",
            "IN": "ADP",
            "DT": "DET",
            "RB": "ADV",
            "RBR": "ADV",
            "RBS": "ADV",
            "CC": "CCONJ",
        }
        return tag_map.get(penn_tag, "OTHER")

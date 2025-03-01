import math
from dataclasses import dataclass

import numpy as np

from src.finders.scorers.base import Scorer


@dataclass
class WordLengthScorer(Scorer):
    """Word length scorer"""

    min_length: int = 1
    length_bonus: float = 0.1  # 10% bonus per character above min_length

    def score_candidates(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool | None = None,
    ) -> np.ndarray:
        scores = np.zeros(len(candidates))
        for i, word in enumerate(candidates):
            scores[i] = math.log1p(
                self.length_bonus * max(0, len(word) - self.min_length)
            )
        return scores


@dataclass
class WordDiscriminator(Scorer):
    """Word discriminator scorer. Scores poorly if a word in the hate-list is present"""

    words: set[str]

    def score_candidates(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool | None = None,
    ) -> np.ndarray:
        scores = np.zeros(len(candidates))
        for i, word in enumerate(candidates):
            scores[i] = -1 if word in self.words else 0
        return scores

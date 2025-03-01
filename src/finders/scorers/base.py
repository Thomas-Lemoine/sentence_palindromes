from typing import Protocol

import numpy as np


class Scorer(Protocol):
    """Base protocol for all scorers"""

    def score_candidates(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool | None = None,
    ) -> np.ndarray:
        """
        Score candidates.
        Returns a numpy array of scores for each candidate.
        """
        ...

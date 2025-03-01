import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.finders.scorers.base import Scorer


@dataclass
class CharacterRepetitionScorer(Scorer):
    """Penalizes words that use already-frequent characters"""

    max_char_frequency_ratio: float = 0.2  # No character should be >20% of total chars

    def score_candidates(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool | None = None,
    ) -> np.ndarray:
        if not candidates:
            return np.array([])

        # Count character frequencies in context
        context_str = "".join(context).lower()
        char_counts = Counter(context_str)
        total_chars = len(context_str) if context_str else 0

        scores = np.zeros(len(candidates))
        for i, word in enumerate(candidates):
            # Create hypothetical new string if we added this word
            new_str = context_str + word.lower()
            new_total = len(new_str)

            # Check frequency ratios in hypothetical new string
            combined_counts = Counter(new_str)
            for char, count in combined_counts.items():
                if count / new_total > self.max_char_frequency_ratio:
                    scores[i] = float("-inf")
                    break

            if scores[i] != float("-inf"):
                # Original scoring logic for non-filtered words
                char_penalties = []
                for char in word.lower():
                    # Penalty scales with current frequency ratio
                    current_ratio = char_counts.get(char, 0) / (total_chars or 1)
                    penalty = -math.log1p(
                        current_ratio * 5
                    )  # Scale factor of 5 to make penalties meaningful
                    char_penalties.append(penalty)
                scores[i] = sum(char_penalties) / len(word) if word else 0.0

        return scores


@dataclass
class RepetitionPenaltyScorer(Scorer):
    max_occurrences: int = 3  # Hard cutoff
    window_size: int = 4  # For recent context

    def score_candidates(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool | None = None,
    ) -> np.ndarray:
        if not candidates:
            return np.array([])

        scores = np.zeros(len(candidates))
        for i, candidate in enumerate(candidates):
            # Hard filter for too many occurrences
            total_count = context.count(candidate)
            if total_count >= self.max_occurrences:
                scores[i] = float("-inf")
                continue

            # Hard filter for immediate repetition
            if context and (
                candidate == context[-1]
                or (len(context) > 1 and candidate == context[-2])
            ):
                scores[i] = float("-inf")
                continue

            # Count recent occurrences
            recent_context = context[-self.window_size :] if context else []
            recent_count = recent_context.count(candidate)

            if recent_count > 0:
                scores[i] = math.log(0.3) * recent_count
            elif total_count > 0:
                scores[i] = math.log(0.5) * total_count

        return scores

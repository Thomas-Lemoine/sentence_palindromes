import math
from dataclasses import dataclass


@dataclass
class WordScore:
    variety_bonus: float = 1.0  # Bonus for using diverse words
    length_bonus: float = 1.0  # Bonus for using longer words
    rarity_bonus: float = 1.0  # Bonus for using less common words

    @classmethod
    def from_words(cls, words: list[str], word_frequencies: dict[str, int]) -> "WordScore":
        if not words:
            return cls()

        # Calculate variety score (penalize repetition)
        unique_ratio = len(set(words)) / len(words)
        variety_bonus = math.pow(unique_ratio, 1.5)  # Exponential bonus for variety

        # Calculate length score (reward longer words)
        avg_length = sum(len(word) for word in words) / len(words)
        length_bonus = math.log2(avg_length + 1)  # Logarithmic bonus for length

        # Calculate rarity score (reward less common words)
        max_freq = max(word_frequencies.values())
        avg_rarity = sum(1 - (word_frequencies.get(w, 0) / max_freq) for w in words) / len(words)
        rarity_bonus = math.pow(avg_rarity + 0.5, 2)  # Quadratic bonus for rarity

        return cls(variety_bonus, length_bonus, rarity_bonus)

    def total_score(self) -> float:
        return self.variety_bonus * self.length_bonus * self.rarity_bonus

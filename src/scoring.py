import math
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ScoringWeights:
    """Weights for different scoring components"""

    variety: float = 1.0  # How much to reward using different words
    word_length: float = 1.0  # How much to reward longer words
    rarity: float = 1.0  # How much to reward uncommon words
    sequence_length: float = 1.0  # How much to reward longer sequences
    syllable_variety: float = 0.5  # How much to reward varied syllable patterns


class FrequencyProvider(Protocol):
    """Protocol for providing word frequency information"""

    def get_frequency(self, word: str) -> int:
        """Get frequency count for a word"""
        ...


class BasicFrequencyProvider:
    """Simple frequency provider using a dictionary"""

    def __init__(self, frequencies: dict[str, int]):
        self.frequencies = frequencies
        self.max_freq = max(frequencies.values()) if frequencies else 1

    def get_frequency(self, word: str) -> int:
        return self.frequencies.get(word, 0)


@dataclass
class PalindromeScore:
    """Detailed scoring breakdown for a palindrome"""

    variety_score: float = 0.0
    word_length_score: float = 0.0
    rarity_score: float = 0.0
    sequence_length_score: float = 0.0
    syllable_score: float = 0.0
    total_score: float = 0.0

    @property
    def components(self) -> dict[str, float]:
        """Get all score components as a dictionary"""
        return {
            "variety": self.variety_score,
            "word_length": self.word_length_score,
            "rarity": self.rarity_score,
            "sequence_length": self.sequence_length_score,
            "syllable": self.syllable_score,
            "total": self.total_score,
        }


class PalindromeScorer:
    """Scorer for palindrome sequences"""

    def __init__(
        self, frequency_provider: FrequencyProvider, weights: ScoringWeights | None = None
    ):
        self.frequency_provider = frequency_provider
        self.weights = weights or ScoringWeights()

    def score_sequence(self, words: list[str]) -> PalindromeScore:
        """Score a sequence of words"""
        if not words:
            return PalindromeScore()

        # Calculate individual components
        variety_score = self._calculate_variety_score(words)
        word_length_score = self._calculate_word_length_score(words)
        rarity_score = self._calculate_rarity_score(words)
        sequence_length_score = self._calculate_sequence_length_score(words)
        syllable_score = self._calculate_syllable_score(words)

        # Combine scores using weights
        total_score = (
            variety_score * self.weights.variety
            + word_length_score * self.weights.word_length
            + rarity_score * self.weights.rarity
            + sequence_length_score * self.weights.sequence_length
            + syllable_score * self.weights.syllable_variety
        ) / sum(vars(self.weights).values())  # Normalize by sum of weights

        return PalindromeScore(
            variety_score=variety_score,
            word_length_score=word_length_score,
            rarity_score=rarity_score,
            sequence_length_score=sequence_length_score,
            syllable_score=syllable_score,
            total_score=total_score,
        )

    def _calculate_variety_score(self, words: list[str]) -> float:
        """Calculate score based on word variety"""
        unique_ratio = len(set(words)) / len(words)
        # Exponential bonus for variety to strongly penalize repetition
        return math.pow(unique_ratio, 1.5)

    def _calculate_word_length_score(self, words: list[str]) -> float:
        """Calculate score based on word lengths"""
        avg_length = sum(len(word) for word in words) / len(words)
        # Logarithmic bonus for length to avoid over-rewarding very long words
        return math.log2(avg_length + 1)

    def _calculate_rarity_score(self, words: list[str]) -> float:
        """Calculate score based on word rarity"""
        max_freq = max(self.frequency_provider.get_frequency(w) for w in words)
        if max_freq == 0:
            return 0.0

        avg_rarity = sum(
            1 - (self.frequency_provider.get_frequency(w) / max_freq) for w in words
        ) / len(words)

        # Quadratic bonus for rarity to reward using uncommon words
        return math.pow(avg_rarity + 0.5, 2)

    def _calculate_sequence_length_score(self, words: list[str]) -> float:
        """Calculate score based on sequence length"""
        # Logarithmic scaling to prefer medium-length sequences
        return math.log2(len(words) + 1) / 3

    def _calculate_syllable_score(self, words: list[str]) -> float:
        """Calculate score based on syllable patterns"""

        # Simple approximation of syllables using vowel groups
        def count_syllables(word: str) -> int:
            vowels = "aeiou"
            count = 0
            in_vowel_group = False
            for char in word.lower():
                is_vowel = char in vowels
                if is_vowel and not in_vowel_group:
                    count += 1
                in_vowel_group = is_vowel
            return max(1, count)

        syllable_counts = [count_syllables(word) for word in words]
        unique_patterns = len(set(syllable_counts))
        return unique_patterns / len(words)


def create_basic_scorer(
    word_frequencies: dict[str, int], weights: ScoringWeights | None = None
) -> PalindromeScorer:
    """Create a basic scorer with given word frequencies"""
    return PalindromeScorer(
        frequency_provider=BasicFrequencyProvider(word_frequencies), weights=weights
    )


# Pre-defined weight configurations
BALANCED_WEIGHTS = ScoringWeights(
    variety=1.0, word_length=1.0, rarity=1.0, sequence_length=1.0, syllable_variety=0.5
)

VARIETY_FOCUSED_WEIGHTS = ScoringWeights(
    variety=2.0, word_length=0.5, rarity=1.0, sequence_length=0.5, syllable_variety=0.3
)

RARITY_FOCUSED_WEIGHTS = ScoringWeights(
    variety=1.0, word_length=0.5, rarity=2.0, sequence_length=0.5, syllable_variety=0.3
)

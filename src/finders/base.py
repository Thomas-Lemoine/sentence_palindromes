import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Iterator, TypeVar

from src.metrics import PalindromeMetrics
from src.vocabulary import get_vocabulary

T = TypeVar("T")


@dataclass
class PalindromeCandidate:
    """Represents a potential palindrome during generation"""

    words: list[str]
    score: float = 0.0

    def __str__(self) -> str:
        return " ".join(self.words)

    @property
    def length(self) -> int:
        return len(self.words)

    def is_valid_length(self, min_length: int, max_length: int) -> bool:
        return min_length <= self.length <= max_length


class PalindromeFinder(ABC):
    """Abstract base class for palindrome finders"""

    def __init__(self, vocabulary: set[str] | None = None):
        if vocabulary is None:
            vocabulary = get_vocabulary()

        self.vocabulary = vocabulary
        # Pre-compute palindromic words for efficiency
        self.palindromic_words = {word for word in vocabulary if self._is_palindrome(word)}

    @staticmethod
    def _is_palindrome(text: str) -> bool:
        """Check if a string is a palindrome"""
        return text == text[::-1]

    def _is_word_sequence_palindrome(self, words: list[str]) -> bool:
        """Check if a sequence of words forms a palindrome"""
        return self._is_palindrome("".join(words))

    @abstractmethod
    def _initialize_search(self) -> Generator[T, None, None]:
        """Initialize the search state(s)

        Returns:
            Generator yielding initial search states
        """
        pass

    @abstractmethod
    def _expand_state(self, state: T) -> Generator[T, None, None]:
        """Generate next possible states from current state

        Args:
            state: Current search state

        Returns:
            Generator yielding next possible states
        """
        pass

    @abstractmethod
    def _state_to_candidate(self, state: T) -> PalindromeCandidate:
        """Convert a search state to a palindrome candidate

        Args:
            state: Search state to convert

        Returns:
            PalindromeCandidate representing the state
        """
        pass

    @abstractmethod
    def _score_candidate(self, candidate: PalindromeCandidate) -> float:
        """Score a palindrome candidate

        Args:
            candidate: Palindrome candidate to score

        Returns:
            Float score for the candidate
        """
        pass

    def generate_palindromes(
        self,
        min_length: int = 3,
        max_length: int = 8,
    ) -> Iterator[tuple[list[str], PalindromeMetrics]]:
        """Generate palindromic sentences.

        Args:
            min_length: Minimum number of words in palindrome
            max_length: Maximum number of words in palindrome

        Yields:
            Tuple of (palindrome word list, current metrics)
        """
        metrics = PalindromeMetrics()
        start_time = time.time()

        # Process each initial state
        states_to_process = list(self._initialize_search())
        seen_sequences = set()  # Track unique sequences we've seen

        while states_to_process:
            current_state = states_to_process.pop()
            candidate = self._state_to_candidate(current_state)

            # Skip if we've seen this sequence
            sequence_key = " ".join(candidate.words)
            if sequence_key in seen_sequences:
                continue
            seen_sequences.add(sequence_key)

            # Check if valid palindrome
            if candidate.is_valid_length(
                min_length, max_length
            ) and self._is_word_sequence_palindrome(candidate.words):
                # Update metrics
                metrics.num_palindromes += 1
                metrics.length_distribution[candidate.length] = (
                    metrics.length_distribution.get(candidate.length, 0) + 1
                )
                metrics.max_length = max(metrics.max_length, candidate.length)

                if metrics.num_palindromes > 0:
                    lengths = list(metrics.length_distribution.keys())
                    counts = list(metrics.length_distribution.values())
                    metrics.avg_length = sum(l * c for l, c in zip(lengths, counts)) / sum(counts)
                metrics.generation_time = time.time() - start_time

                yield candidate.words, metrics

            # If not at max length, expand state
            if candidate.length < max_length:
                states_to_process.extend(self._expand_state(current_state))

    def has_repeating_pattern(self, words: list[str], min_repetitions: int = 3) -> bool:
        """Check if a sequence has repeating patterns

        Args:
            words: List of words to check
            min_repetitions: Minimum number of repetitions to consider a pattern

        Returns:
            True if repeating pattern found, False otherwise
        """
        n = len(words)
        # Check patterns of different lengths
        for length in range(1, n // min_repetitions + 1):
            # Check different starting positions
            for start in range(n - (length * (min_repetitions - 1))):
                pattern = words[start : start + length]
                # Count repetitions
                count = 1
                pos = start + length
                while pos + length <= n:
                    if words[pos : pos + length] == pattern:
                        count += 1
                        if count >= min_repetitions:
                            return True
                        pos += length
                    else:
                        break
        return False

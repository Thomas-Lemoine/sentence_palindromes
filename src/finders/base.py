from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np

from src.finders.scorers.base import Scorer


class WordFilterStrategy(Protocol):
    """Protocol for filtering word candidates"""

    def should_keep_word(self, context: list[str], candidate: str) -> bool:
        """Return True if candidate word should be considered"""
        ...


@dataclass
class PalindromeFinder:
    vocabulary: set[str]
    prefix_cache: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    suffix_cache: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # Strategy fields
    scorers: list[Scorer] = field(default_factory=list)
    branching_factor: int = 5

    def __post_init__(self) -> None:
        """Initialize caches after instance creation"""
        self._build_caches()

    def _build_caches(self) -> None:
        """Build prefix and suffix caches for efficient word lookup"""
        for word in self.vocabulary:
            for i in range(1, len(word) + 1):
                prefix, suffix = word[:i], word[-i:]
                self.prefix_cache[prefix].add(word)
                self.suffix_cache[suffix].add(word)

    @staticmethod
    def is_palindrome(words: list[str]) -> bool:
        """Check if word sequence forms palindrome"""
        s = "".join(words)
        return s == s[::-1]

    def find_matches(self, pattern: str, match_start: bool = True) -> set[str]:
        """Find all words that start/end with pattern"""
        return (
            self.prefix_cache.get(pattern, set())
            if match_start
            else self.suffix_cache.get(pattern, set())
        )

    def find_mismatch(self, words: list[str], center_pos: int) -> tuple[str, bool]:
        """
        Find what needs to be matched and on which side.
        Returns (unmatched_portion, needs_right_match)
        """
        s = "".join(words)
        left, right = s[:center_pos], s[center_pos + 1 :]

        # Find length of matching portion
        match_len = 0
        for i in range(min(len(left), len(right))):
            if left[-(i + 1)] != right[i]:
                break
            match_len = i + 1

        # Get unmatched portions
        left_unmatched = left[:-match_len] if match_len else left
        right_unmatched = right[match_len:] if match_len else right

        # Return longer unmatched portion and whether we need to match on right
        return (
            (left_unmatched, True)
            if len(left_unmatched) >= len(right_unmatched)
            else (right_unmatched, False)
        )

    def score(
        self,
        context: list[str],
        candidates: list[str],
        adding_right: bool,
    ) -> list[float]:
        """Returns log-odds scores for each candidate"""
        scores = np.zeros(len(candidates))
        for scorer in self.scorers:
            scores += scorer.score_candidates(context, candidates, adding_right)
        return list(scores)

    def filter_candidates(
        self,
        words: list[str],
        candidates: list[str],
        adding_right: bool,
    ) -> set[str]:
        """Filter candidates based on scorers and return top
        N candidates by score that are not -inf scored."""
        if not candidates:
            return set()
        filtered = candidates.copy()
        scores = self.score(words, filtered, adding_right)

        # Create list of (candidate, score) tuples, filtering out -inf scores
        scored_candidates = [
            (candidate, score)
            for candidate, score in zip(filtered, scores)
            if score != float("-inf")
        ]

        # Sort by score in descending order
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Extract just the candidates (without scores)
        filtered_candidates = [c for c, _ in scored_candidates]

        return set(filtered_candidates[: self.branching_factor])

    def grow_palindromes(self, words: list[str], center_pos: int, depth: int = 5) -> Iterator[str]:
        """
        Recursively grow palindromes from initial words.
        Yields valid palindromes as space-separated strings.
        """
        if depth <= 0:
            return

        # First check if current sequence is a palindrome
        if self.is_palindrome(words):
            yield " ".join(words)
            # Don't return - continue growing to find longer palindromes  TODO: VERIFY

        mismatch, needs_right = self.find_mismatch(words, center_pos)
        if not mismatch:
            return

        # Find matching words for the reversed mismatch pattern
        pattern = mismatch[::-1]
        matches = self.find_matches(pattern, match_start=needs_right)

        filtered_matches = self.filter_candidates(words, matches, needs_right)

        for word in filtered_matches:
            new_words = words + [word] if needs_right else [word] + words
            new_center = center_pos if needs_right else center_pos + len(word)

            if self.is_palindrome(new_words):
                yield " ".join(new_words)
            yield from self.grow_palindromes(new_words, new_center, depth - 1)

    def generate_palindromes(
        self, depth: int = 5, custom_centers: list[tuple[str, int]] | None = None
    ) -> Iterator[str]:
        """
        Generate palindromes from all potential centers found in vocabulary.
        Also includes starting from meaningful seed sequences like ["be"].
        """
        for word, center_pos in custom_centers or []:
            yield from self.grow_palindromes([word], center_pos, depth)

        for word, center_pos in self.find_palindrome_centers():
            yield from self.grow_palindromes([word], center_pos, depth)

    def find_palindrome_centers(self) -> list[tuple[str, int]]:
        """
        Find all potential palindrome centers in vocabulary words.
        Returns list of (word, position) tuples where position could be center of palindrome.
        Only returns positions where there's a true palindrome opportunity.
        """
        results = []

        for word in self.vocabulary:
            length = len(word)
            if length < 3:  # Skip very short words
                continue

            # For each position (excluding first two and last two characters)
            for pos in range(1, length - 1):
                # Skip center position of word
                if pos == length // 2:
                    continue

                # Get entire left and right substrings
                left = word[:pos]
                right = word[pos + 1 :]

                # Check if left matches reverse of right (up to shorter length)
                min_length = min(len(left), len(right))
                if min_length > 0 and left[-min_length:] == right[:min_length][::-1]:
                    results.append((word, pos))

        return results

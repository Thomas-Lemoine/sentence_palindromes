from itertools import product
from typing import Generator

from src.finders.base import PalindromeCandidate, PalindromeFinder
from src.utils import is_word_sequence_palindrome


class BruteForceFinder(PalindromeFinder):
    """Palindrome finder using brute force search through combinations"""

    def __init__(self):
        super().__init__()
        self.compatible_suffixes = self._precompute_suffixes()

    def _precompute_suffixes(self) -> dict[str, list[str]]:
        """For each word, find all words that could appear at the end of a palindrome starting with it"""
        compatible = {}
        for w1 in self.vocabulary:
            rev_w1 = w1[::-1]
            compatible[w1] = [
                w2
                for w2 in self.vocabulary
                if w2.endswith(rev_w1) or rev_w1.endswith(w2)
            ]
        return compatible

    def _initialize_search(self) -> Generator[PalindromeCandidate, None, None]:
        """Initialize with palindromic single words"""
        yield from (
            PalindromeCandidate([word])
            for word in self.vocabulary
            if word in self.palindromic_words
        )

    def _expand_state(
        self, state: PalindromeCandidate
    ) -> Generator[PalindromeCandidate, None, None]:
        """Generate all possible expansions by trying combinations"""
        if not state.words:
            return

        # Current first and last words
        first_word = state.words[0]

        # For single word states, try all compatible end words
        if len(state.words) == 1:
            for last_word in self.compatible_suffixes[first_word]:
                sequence = [first_word, last_word]
                if is_word_sequence_palindrome(sequence):
                    yield PalindromeCandidate(sequence)
            return

        # For longer states, try all possible middle word combinations
        last_word = state.words[-1]
        length = len(state.words)

        # Generate all possible middle word combinations
        middle_combinations = product(self.vocabulary, repeat=length)
        for middle in middle_combinations:
            sequence = [first_word] + list(middle) + [last_word]
            if is_word_sequence_palindrome(sequence):
                yield PalindromeCandidate(sequence)

    def _state_to_candidate(self, state: PalindromeCandidate) -> PalindromeCandidate:
        """State is already a candidate in this implementation"""
        return state

    def _score_candidate(self, candidate: PalindromeCandidate) -> float:
        """Score using the common scoring system"""
        return self.scorer.score_sequence(candidate.words).total_score


def main():
    finder = BruteForceFinder()

    print(f"Vocabulary size: {len(finder.vocabulary)}")
    print(f"Palindromic words: {len(finder.palindromic_words)}")
    print()

    print("Generating palindromes...")
    for palindrome, metrics in finder.generate_palindromes(min_length=3, max_length=6):
        print(f"Found: {' '.join(palindrome)}")
        if metrics.num_palindromes % 100 == 0:
            print(f"\nCurrent Metrics:\n{metrics}\n")


if __name__ == "__main__":
    main()

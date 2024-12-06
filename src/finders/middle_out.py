import time
from typing import Generator

from src.finders.base import PalindromeCandidate, PalindromeFinder
from src.utils import has_repeating_pattern, precompute_compatible_pairs


class MiddleOutFinder(PalindromeFinder):
    """Middle-out implementation of palindrome finding"""

    def __init__(self):
        super().__init__()
        self.compatible_pairs = precompute_compatible_pairs(self.vocabulary)

    def _initialize_search(self) -> Generator[PalindromeCandidate, None, None]:
        """Start with empty sequence and palindromic words"""
        yield PalindromeCandidate([])  # Empty center
        for word in self.palindromic_words:
            yield PalindromeCandidate([word])

    def _expand_state(
        self, state: PalindromeCandidate
    ) -> Generator[PalindromeCandidate, None, None]:
        """Expand state by trying different length combinations"""
        sequence = state.words

        for word1 in self.vocabulary:
            # Try adding just one word that makes it a palindrome
            new_sequence = [word1] + sequence
            if self._is_palindrome(" ".join(new_sequence)):
                yield PalindromeCandidate(new_sequence)

            # Try adding word pairs
            for word2 in self.vocabulary:
                new_sequence = [word1] + sequence + [word2]
                if self._is_palindrome(" ".join(new_sequence)):
                    yield PalindromeCandidate(new_sequence)

    def _state_to_candidate(self, state: PalindromeCandidate) -> PalindromeCandidate:
        """State is already a candidate in this implementation"""
        return state

    def _score_candidate(self, candidate: PalindromeCandidate) -> float:
        """Score using the common scoring system"""
        return self.scorer.score_sequence(candidate.words).total_score


def main():
    finder = MiddleOutFinder()

    print(f"Vocabulary size: {len(finder.vocabulary)}")
    print(f"Palindromic words: {len(finder.palindromic_words)}")
    print(f"Compatible pairs: {sum(len(pairs) for pairs in finder.compatible_pairs.values())}")
    print()

    print("Generating palindromes...")
    start_time = time.time()

    for palindrome, metrics in finder.generate_palindromes(min_length=3, max_length=8):
        print(f"Found: {' '.join(palindrome)}")

        # Print metrics every 100 palindromes
        if metrics.num_palindromes % 100 == 0:
            print(f"\nCurrent Metrics:\n{metrics}\n")

        # Stop after 5 minutes to avoid infinite loops during testing
        if time.time() - start_time > 300:
            print("\nStopping after 5 minutes")
            break


if __name__ == "__main__":
    main()

from heapq import nlargest
from typing import Generator

from src.finders.base import PalindromeCandidate, PalindromeFinder
from src.utils import (
    has_repeating_pattern,
    is_word_sequence_palindrome,
    precompute_compatible_pairs,
)


class BeamSearchFinder(PalindromeFinder):
    """Palindrome finder using beam search with prioritized expansion"""

    def __init__(self, beam_width: int = 1000):
        super().__init__()
        self.beam_width = beam_width
        self.compatible_pairs = precompute_compatible_pairs(self.vocabulary)

    def _initialize_search(self) -> Generator[PalindromeCandidate, None, None]:
        """Start with empty sequence and palindromic words"""
        yield PalindromeCandidate([])  # Empty center
        for word in self.palindromic_words:
            yield PalindromeCandidate([word])

    def _expand_state(
        self, state: PalindromeCandidate
    ) -> Generator[PalindromeCandidate, None, None]:
        """Expand state using beam search"""
        sequence = state.words

        if not sequence:
            # For empty sequences, use pre-computed compatible pairs
            for word1, compatible in self.compatible_pairs.items():
                for word2 in compatible:
                    new_sequence = [word1, word2]
                    if not has_repeating_pattern(new_sequence):
                        candidate = PalindromeCandidate(new_sequence)
                        candidate.score = self._score_candidate(candidate)
                        yield candidate
        else:
            # Try extending with words that maintain palindrome property
            expansions = []
            for word1 in self.vocabulary:
                if sequence and word1 == sequence[0]:  # Skip immediate repetition
                    continue

                for word2 in self.vocabulary:
                    if sequence and word2 == sequence[-1]:  # Skip immediate repetition
                        continue

                    new_sequence = [word1] + sequence + [word2]
                    if is_word_sequence_palindrome(new_sequence) and not has_repeating_pattern(
                        new_sequence
                    ):
                        candidate = PalindromeCandidate(new_sequence)
                        candidate.score = self._score_candidate(candidate)
                        expansions.append(candidate)

            # Return top-k expansions based on score
            for candidate in nlargest(self.beam_width, expansions, key=lambda x: x.score):
                yield candidate

    def _state_to_candidate(self, state: PalindromeCandidate) -> PalindromeCandidate:
        """State is already a candidate in this implementation"""
        return state

    def _score_candidate(self, candidate: PalindromeCandidate) -> float:
        """Score using the common scoring system"""
        return self.scorer.score_sequence(candidate.words).total_score


def main():
    finder = BeamSearchFinder(beam_width=1000)

    print(f"Vocabulary size: {len(finder.vocabulary)}")
    print(f"Palindromic words: {len(finder.palindromic_words)}")
    print(f"Compatible pairs: {sum(len(pairs) for pairs in finder.compatible_pairs.values())}")
    print()

    print("Generating palindromes...")
    for palindrome, metrics in finder.generate_palindromes(min_length=3, max_length=8):
        print(f"Found: {' '.join(palindrome)}")
        if metrics.num_palindromes % 100 == 0:
            print(f"\nCurrent Metrics:\n{metrics}\n")


if __name__ == "__main__":
    main()

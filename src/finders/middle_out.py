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
        """Expand state by adding compatible word pairs to ends"""
        sequence = state.words

        if not sequence:
            # For empty sequences, use pre-computed compatible pairs
            for word1, compatible in self.compatible_pairs.items():
                for word2 in compatible:
                    new_sequence = [word1, word2]
                    if not has_repeating_pattern(new_sequence):
                        yield PalindromeCandidate(new_sequence)
        else:
            # Try extending with words that maintain palindrome property
            text = "".join(sequence)
            for word1 in self.vocabulary:
                # Skip if this would create immediate repetition
                if sequence and word1 == sequence[0]:
                    continue

                for word2 in self.vocabulary:
                    # Skip if this would create immediate repetition
                    if sequence and word2 == sequence[-1]:
                        continue

                    new_sequence = [word1] + sequence + [word2]
                    if self._is_word_sequence_palindrome(
                        new_sequence
                    ) and not has_repeating_pattern(new_sequence):
                        yield PalindromeCandidate(new_sequence)

    def _state_to_candidate(self, state: PalindromeCandidate) -> PalindromeCandidate:
        """State is already a candidate in this implementation"""
        return state

    def _score_candidate(self, candidate: PalindromeCandidate) -> float:
        """Score based on length, variety, and word lengths"""
        if not candidate.words:
            return 0.0

        # Reward variety
        variety_score = len(set(candidate.words)) / len(candidate.words)

        # Reward longer words
        avg_word_length = sum(len(word) for word in candidate.words) / len(candidate.words)
        length_score = avg_word_length / 10  # Normalize to roughly 0-1 range

        # Reward sequence length
        sequence_score = len(candidate.words) / 8  # Normalize to roughly 0-1 range

        return (variety_score + length_score + sequence_score) / 3


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

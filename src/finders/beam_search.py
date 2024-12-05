from heapq import nlargest
from typing import Generator

from src.finders.base import PalindromeCandidate, PalindromeFinder
from src.utils import has_repeating_pattern, is_word_sequence_palindrome


class BeamSearchFinder(PalindromeFinder):
    """Palindrome finder using beam search with prioritized expansion"""

    def __init__(self, beam_width: int = 1000):
        super().__init__()
        self.beam_width = beam_width
        self.compatible_pairs = self._precompute_compatible_pairs()

    def _precompute_compatible_pairs(self) -> dict[str, set[str]]:
        """Find all pairs of words that could form palindromes together"""
        pairs = {}
        for word1 in self.vocabulary:
            rev_word1 = word1[::-1]
            compatible = set()
            for word2 in self.vocabulary:
                if word1 != word2:  # Avoid same word
                    if self._is_palindrome(word1 + word2):
                        compatible.add(word2)
            if compatible:
                pairs[word1] = compatible
        return pairs

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

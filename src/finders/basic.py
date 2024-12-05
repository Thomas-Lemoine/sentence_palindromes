from typing import Generator

from src.finders.base import PalindromeCandidate, PalindromeFinder
from src.trie import Trie
from src.utils import has_repeating_pattern, is_word_sequence_palindrome


class BasicFinder(PalindromeFinder):
    """Basic palindrome finder using tries and compatible pair matching"""

    def __init__(self):
        super().__init__()
        self.forward_trie = Trie()
        self.reverse_trie = Trie()
        self._build_tries()
        self.compatible_pairs = self._precompute_compatible_pairs()

    def _build_tries(self) -> None:
        """Build forward and reverse tries from vocabulary"""
        for word in self.vocabulary:
            self.forward_trie.insert(word)
            self.reverse_trie.insert(word[::-1])

    def _initialize_search(self) -> Generator[PalindromeCandidate, None, None]:
        """Start with empty center and palindromic words"""
        yield PalindromeCandidate([])  # Empty center
        for word in self.palindromic_words:
            yield PalindromeCandidate([word])

    def _get_valid_extensions(self, sequence: list[str]) -> Generator[tuple[str, str], None, None]:
        """Find valid word pairs to extend the sequence"""
        if not sequence:
            # For empty sequences, use pre-computed compatible pairs
            for word1 in self.vocabulary:
                for word2 in self.compatible_pairs.get(word1, set()):
                    if word1 != word2:  # Avoid same word pairs
                        if not has_repeating_pattern([], (word1, word2)):
                            yield word1, word2
        else:
            # Try extending with words that maintain palindrome property
            for word1 in self.vocabulary:
                # Skip if this would create immediate repetition
                if sequence and word1 == sequence[0]:
                    continue

                rev_suffix = word1[::-1]
                possible_matches = self.reverse_trie.find_words_with_prefix(rev_suffix)

                for word2 in possible_matches:
                    word2_rev = word2[::-1]
                    # Skip if this would create immediate repetition
                    if sequence and word2_rev == sequence[-1]:
                        continue

                    new_sequence = [word1] + sequence + [word2_rev]
                    if is_word_sequence_palindrome(new_sequence) and not has_repeating_pattern(
                        sequence, (word1, word2_rev)
                    ):
                        yield word1, word2_rev

    def _expand_state(
        self, state: PalindromeCandidate
    ) -> Generator[PalindromeCandidate, None, None]:
        """Expand state by adding valid word pairs to ends"""
        for left, right in self._get_valid_extensions(state.words):
            yield PalindromeCandidate([left] + state.words + [right])

    def _state_to_candidate(self, state: PalindromeCandidate) -> PalindromeCandidate:
        """State is already a candidate in this implementation"""
        return state

    def _score_candidate(self, candidate: PalindromeCandidate) -> float:
        """Score based on length and variety"""
        if not candidate.words:
            return 0.0

        # Reward variety
        variety_score = len(set(candidate.words)) / len(candidate.words)
        # Reward word length
        avg_word_length = sum(len(word) for word in candidate.words) / len(candidate.words)
        length_score = avg_word_length / 10

        return (variety_score + length_score) / 2


def main():
    finder = BasicFinder()

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

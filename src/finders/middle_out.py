from dataclasses import dataclass
from typing import Generator

from src.finders.base import BasePalindromeFinder, PalindromeCandidate


class SimpleMiddleOutFinder(BasePalindromeFinder):
    """Simple middle-out implementation for demonstration"""

    @dataclass
    class SearchState:
        """Search state for middle-out approach"""

        sequence: list[str]

    def _initialize_search(self) -> Generator[SearchState, None, None]:
        """Start with empty sequence and palindromic words"""
        yield self.SearchState([])  # Empty center
        for word in self.palindromic_words:  # Single palindromic words
            yield self.SearchState([word])

    def _expand_state(self, state: SearchState) -> Generator[SearchState, None, None]:
        """Expand by adding word pairs that maintain palindrome property"""
        sequence = state.sequence
        # Try adding word pairs to ends
        for word1 in self.vocabulary:
            for word2 in self.vocabulary:
                new_sequence = [word1] + sequence + [word2]
                if self._is_word_sequence_palindrome(new_sequence):
                    if not self.has_repeating_pattern(new_sequence):
                        yield self.SearchState(new_sequence)

    def _state_to_candidate(self, state: SearchState) -> PalindromeCandidate:
        """Convert search state to candidate"""
        return PalindromeCandidate(
            words=state.sequence, score=self._score_candidate(PalindromeCandidate(state.sequence))
        )

    def _score_candidate(self, candidate: PalindromeCandidate) -> float:
        """Simple scoring based on length and variety"""
        if not candidate.words:
            return 0.0
        variety_score = len(set(candidate.words)) / len(candidate.words)
        length_score = sum(len(word) for word in candidate.words) / len(candidate.words)
        return variety_score * length_score


if __name__ == "__main__":
    finder = SimpleMiddleOutFinder()
    for state in finder._initialize_search():
        print(state)

import time
from collections import defaultdict
from typing import Dict, Iterator, List, Set, Tuple

import nltk
from nltk.corpus import brown

from src.metrics import PalindromeMetrics
from src.trie import Trie


class PalindromeFinder:
    def __init__(self, vocabulary: Set[str]):
        self.vocabulary = vocabulary
        self.forward_trie = Trie()
        self.reverse_trie = Trie()
        self._build_tries()

        # Pre-compute word pairs that could be valid extensions
        self.compatible_pairs = self._precompute_compatible_pairs()

    def _build_tries(self) -> None:
        """Build forward and reverse tries from vocabulary"""
        for word in self.vocabulary:
            self.forward_trie.insert(word)
            self.reverse_trie.insert(word[::-1])

    def _precompute_compatible_pairs(self) -> Dict[str, Set[str]]:
        """Precompute pairs of words that could be valid palindrome extensions"""
        pairs = defaultdict(set)
        for word1 in self.vocabulary:
            rev_word1 = word1[::-1]
            for word2 in self.vocabulary:
                if word1 != word2:  # Avoid same word
                    combined = word1 + word2
                    rev_combined = combined[::-1]
                    if combined == rev_combined:
                        pairs[word1].add(word2)
        return pairs

    def _is_palindrome(self, words: List[str]) -> bool:
        """Check if a sequence of words forms a palindrome"""
        text = "".join(words)
        return text == text[::-1]

    def _get_valid_extensions(self, center: List[str]) -> Iterator[Tuple[str, str]]:
        """Get valid word pairs that could extend the current palindrome"""
        if not center:
            # For empty center, any palindrome pair works
            for word1 in self.vocabulary:
                for word2 in self.compatible_pairs[word1]:
                    yield word1, word2
        else:
            # Get prefix and suffix constraints from current palindrome
            text = "".join(center)
            for word1 in self.vocabulary:
                combined = word1 + text
                rev_suffix = word1[::-1]
                possible_matches = self.reverse_trie.find_words_with_prefix(rev_suffix)
                for word2 in possible_matches:
                    if self._is_palindrome(center + [word1, word2[::-1]]):
                        yield word1, word2[::-1]

    def generate_palindromes(
        self, min_length: int = 3, max_length: int = 8
    ) -> Iterator[Tuple[List[str], PalindromeMetrics]]:
        """Generate palindromic sentences using middle-out approach"""
        metrics = PalindromeMetrics()
        start_time = time.time()

        # Try different center configurations
        centers = [
            [],  # Empty center for even-length palindromes
            *[[word] for word in self.vocabulary if word == word[::-1]],  # Single palindromic words
        ]

        for center in centers:
            stack = [(center, [])]  # (current_sequence, path_taken)
            while stack:
                sequence, path = stack.pop()

                # If sequence length is within bounds and forms palindrome, yield it
                if min_length <= len(sequence) <= max_length and self._is_palindrome(sequence):
                    metrics.num_palindromes += 1
                    metrics.length_distribution[len(sequence)] = (
                        metrics.length_distribution.get(len(sequence), 0) + 1
                    )
                    metrics.max_length = max(metrics.max_length, len(sequence))

                    # Update metrics
                    lengths = list(metrics.length_distribution.keys())
                    counts = list(metrics.length_distribution.values())
                    metrics.avg_length = sum(l * c for l, c in zip(lengths, counts)) / sum(counts)
                    metrics.generation_time = time.time() - start_time

                    yield sequence, metrics

                # If we haven't reached max length, try extending
                if len(sequence) < max_length:
                    for left, right in self._get_valid_extensions(sequence):
                        new_sequence = [left] + sequence + [right]
                        new_path = path + [(left, right)]
                        stack.append((new_sequence, new_path))


def get_vocabulary(top_n: int = 5000, min_word_len: int = 2) -> Set[str]:
    """Get vocabulary from Brown corpus"""
    nltk.download("brown", quiet=True)
    freq_dist = nltk.FreqDist(
        word.lower() for word in brown.words() if word.isalpha() and len(word) >= min_word_len
    )
    return {word for word, _ in freq_dist.most_common(top_n)}


if __name__ == "__main__":
    # Configure parameters
    VOCAB_SIZE = 5000
    MIN_LENGTH = 3
    MAX_LENGTH = 8

    # Initialize
    vocabulary = get_vocabulary(VOCAB_SIZE)
    finder = PalindromeFinder(vocabulary)

    # Generate palindromes
    print(f"Generating palindromes (length {MIN_LENGTH}-{MAX_LENGTH})...")
    for palindrome, metrics in finder.generate_palindromes(MIN_LENGTH, MAX_LENGTH):
        print(f"Found: {' '.join(palindrome)}")
        if metrics.num_palindromes % 100 == 0:  # Print metrics periodically
            print(f"\nCurrent Metrics:\n{metrics}\n")

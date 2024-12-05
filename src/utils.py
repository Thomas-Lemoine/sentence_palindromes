from collections import defaultdict
from typing import Dict, List, Set

import nltk
from nltk.corpus import brown


def get_vocabulary(top_n: int = 5000, min_word_len: int = 2) -> set[str]:
    """Get vocabulary from Brown corpus"""
    nltk.download("brown", quiet=True)
    freq_dist = nltk.FreqDist(
        word.lower() for word in brown.words() if word.isalpha() and len(word) >= min_word_len
    )
    return {word for word, _ in freq_dist.most_common(top_n)}


def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome"""
    return text == text[::-1]


def is_word_sequence_palindrome(words: List[str]) -> bool:
    """Check if a sequence of words forms a palindrome"""
    return is_palindrome("".join(words))


def find_palindromic_words(vocabulary: Set[str]) -> Set[str]:
    """Find all palindromic words in vocabulary"""
    return {word for word in vocabulary if is_palindrome(word)}


def precompute_compatible_pairs(vocabulary: Set[str]) -> Dict[str, Set[str]]:
    """Find all pairs of words that could form palindromes together"""
    pairs = defaultdict(set)
    for word1 in vocabulary:
        rev_word1 = word1[::-1]
        for word2 in vocabulary:
            if word1 != word2:  # Avoid same word
                if is_palindrome(word1 + word2):
                    pairs[word1].add(word2)
    return pairs


def has_repeating_pattern(words: List[str], min_repetitions: int = 3) -> bool:
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

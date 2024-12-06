from collections import defaultdict

import nltk
from nltk.corpus import brown


def get_vocabulary(top_n: int = 5000, min_word_len: int = 2) -> set[str]:
    """Get vocabulary from Brown corpus"""
    try:
        nltk.data.find("brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    freq_dist = nltk.FreqDist(
        word.lower() for word in brown.words() if word.isalpha() and len(word) >= min_word_len
    )
    return {word for word, _ in freq_dist.most_common(top_n)}


def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome at character level, ignoring case and spaces"""
    # Only strip spaces and lowercase when checking palindrome property
    cleaned = "".join(c.lower() for c in text if not c.isspace())
    return cleaned == cleaned[::-1]


def is_word_sequence_palindrome(words: list[str]) -> bool:
    """Check if a sequence of words forms a palindrome, ignoring spaces"""
    return is_palindrome(" ".join(words))


def find_palindromic_words(vocabulary: set[str]) -> set[str]:
    """Find all palindromic words in vocabulary"""
    return {word for word in vocabulary if is_palindrome(word)}


def precompute_compatible_pairs(vocabulary: set[str]) -> dict[str, set[str]]:
    """Find all pairs of words that could form palindromes together"""
    pairs = defaultdict(set)
    for word1 in vocabulary:
        rev_word1 = word1[::-1]
        for word2 in vocabulary:
            if word1 != word2:  # Avoid same word
                if is_palindrome(word1 + word2):
                    pairs[word1].add(word2)
    return pairs


def has_repeating_pattern(
    words: list[str], new_words: tuple[str, str] | None = None, min_repetitions: int = 3
) -> bool:
    """Check if a sequence has repeating patterns

    Args:
        words: List of words to check
        new_words: Optional tuple of (left, right) words to be added
        min_repetitions: Minimum number of repetitions to consider a pattern

    Returns:
        True if repeating pattern found, False otherwise
    """
    # If new_words provided, check the extended sequence
    if new_words is not None:
        sequence = [new_words[0]] + words + [new_words[1]]
    else:
        sequence = words

    n = len(sequence)
    if n < min_repetitions:  # Can't have repeating pattern if sequence is too short
        return False

    # Check patterns of different lengths
    for length in range(1, n // min_repetitions + 1):
        # Check different starting positions
        for start in range(n - (length * (min_repetitions - 1))):
            pattern = sequence[start : start + length]
            # Count repetitions
            count = 1
            pos = start + length
            while pos + length <= n:
                if sequence[pos : pos + length] == pattern:
                    count += 1
                    if count >= min_repetitions:
                        return True
                    pos += length
                else:
                    break
    return False

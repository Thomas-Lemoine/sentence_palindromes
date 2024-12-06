from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class PalindromeFinder:
    vocabulary: set[str]
    prefix_cache: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    suffix_cache: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def __post_init__(self) -> None:
        """Initialize caches after instance creation"""
        self._build_caches()

    def _build_caches(self) -> None:
        """Build prefix and suffix caches for efficient word lookup"""
        for word in self.vocabulary:
            for i in range(1, len(word) + 1):
                prefix, suffix = word[:i], word[-i:]
                self.prefix_cache[prefix].add(word)
                self.suffix_cache[suffix].add(word)

    @staticmethod
    def is_palindrome(words: list[str]) -> bool:
        """Check if word sequence forms palindrome"""
        s = "".join(words)
        return s == s[::-1]

    def find_matches(self, pattern: str, match_start: bool = True) -> set[str]:
        """Find all words that start/end with pattern"""
        return (
            self.prefix_cache.get(pattern, set())
            if match_start
            else self.suffix_cache.get(pattern, set())
        )

    def find_mismatch(self, words: list[str], center_pos: int) -> tuple[str, bool]:
        """
        Find what needs to be matched and on which side.
        Returns (unmatched_portion, needs_right_match)
        """
        s = "".join(words)
        left, right = s[:center_pos], s[center_pos + 1 :]

        # Find length of matching portion
        match_len = 0
        for i in range(min(len(left), len(right))):
            if left[-(i + 1)] != right[i]:
                break
            match_len = i + 1

        # Get unmatched portions
        left_unmatched = left[:-match_len] if match_len else left
        right_unmatched = right[match_len:] if match_len else right

        # Return longer unmatched portion and whether we need to match on right
        return (
            (left_unmatched, True)
            if len(left_unmatched) >= len(right_unmatched)
            else (right_unmatched, False)
        )

    def grow_palindromes(self, words: list[str], center_pos: int, depth: int = 5) -> Iterator[str]:
        """
        Recursively grow palindromes from initial words.
        Yields valid palindromes as space-separated strings.
        """
        if depth <= 0:
            return

        if self.is_palindrome(words):
            yield " ".join(words)
            return

        mismatch, needs_right = self.find_mismatch(words, center_pos)
        if not mismatch:
            return

        # Find matching words for the reversed mismatch pattern
        pattern = mismatch[::-1]
        matches = self.find_matches(pattern, match_start=needs_right)

        for word in matches:
            new_words = words + [word] if needs_right else [word] + words
            new_center = center_pos if needs_right else center_pos + len(word)

            if self.is_palindrome(new_words):
                yield " ".join(new_words)
            yield from self.grow_palindromes(new_words, new_center, depth - 1)


def filter_palindrome(palindrome: str, min_avg_length: int = 5) -> bool:
    """
    Filter palindromes based on criteria:
    - No repeated words
    - Minimum average word length
    - First word doesn't contain palindrome prefix
    """
    words = palindrome.split()

    # Check for repeated words
    if len(set(words)) != len(words):
        return False

    # Check average word length
    if sum(len(word) for word in words) / len(words) < min_avg_length:
        return False

    # Check first word for palindrome prefix
    first_word = words[0]
    return not any(PalindromeFinder.is_palindrome([first_word[:i]]) for i in range(2, 7))


def main():
    from src.utils import get_vocabulary

    vocabulary = get_vocabulary(top_n=50000)
    finder = PalindromeFinder(vocabulary)

    print("Finding palindromes...")
    for palindrome in finder.grow_palindromes(["be"], center_pos=0, depth=6):
        if filter_palindrome(palindrome):
            print(palindrome)


if __name__ == "__main__":
    main()

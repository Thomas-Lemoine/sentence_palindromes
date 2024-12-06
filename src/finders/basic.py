from collections import Counter

from src.finders.base import PalindromeFinder, SearchState


class BasicFinder(PalindromeFinder):
    """Basic palindrome finder with minimal pruning"""

    def __init__(self, vocabulary: set[str], max_repeats: int = 1):
        super().__init__(vocabulary)
        self.max_repeats = max_repeats

    def _should_prune(self, state: SearchState) -> bool:
        """Only prune on repeated words"""
        counts = Counter(state.words)
        return any(count > self.max_repeats for count in counts.values())

    def _filter_palindrome(self, state: SearchState) -> bool:
        """Accept all valid palindromes"""
        return True


def main():
    from src.utils import get_vocabulary

    vocabulary = get_vocabulary()
    finder = BasicFinder(vocabulary)

    print("Generating palindromes...")
    count = 0
    for palindrome in finder.generate_palindromes(max_depth=10):
        print(f"  {' '.join(palindrome)}")
        count += 1
        if count > 100:
            print("\nFound over 100 palindromes, stopping...")
            break


if __name__ == "__main__":
    main()

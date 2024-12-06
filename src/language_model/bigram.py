import math
from collections import defaultdict

from src.language_model.base import LanguageModel


class BigramLanguageModel(LanguageModel):
    """Simple bigram language model with add-k smoothing"""

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k
        self._vocab: set[str] = set()
        self._word_counts: dict[str, int] = defaultdict(int)
        self._bigram_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._total_bigrams = 0

    def train(self, corpus: list[list[str]]) -> None:
        """Train on corpus using bigram counts with add-k smoothing"""
        # Collect counts
        for sentence in corpus:
            sentence = [w.lower() for w in sentence]

            # Count words
            for word in sentence:
                self._vocab.add(word)
                self._word_counts[word] += 1

            # Count bigrams
            for w1, w2 in zip(sentence[:-1], sentence[1:]):
                self._bigram_counts[(w1, w2)] += 1
                self._total_bigrams += 1

        # Update stats
        self.stats.unique_words = len(self._vocab)
        self.stats.total_words = sum(self._word_counts.values())
        self.stats.unique_bigrams = len(self._bigram_counts)
        self.stats.total_bigrams = self._total_bigrams
        self.stats.vocab_coverage = len(self._vocab) / len(set(w for s in corpus for w in s))

    def score_transition(self, word1: str, word2: str) -> float:
        """Score transition using smoothed bigram probability"""
        word1, word2 = word1.lower(), word2.lower()

        # If either word not in vocab, return 0
        if word1 not in self._vocab or word2 not in self._vocab:
            return 0.0

        # Get counts with smoothing
        bigram_count = self._bigram_counts.get((word1, word2), 0) + self.k
        word1_count = self._word_counts[word1] + (self.k * len(self._vocab))

        # Calculate probability
        prob = bigram_count / word1_count
        return prob

    def score_sequence(self, words: list[str]) -> float:
        """Score sequence using product of transition probabilities"""
        if len(words) < 2:
            return 1.0 if words and words[0].lower() in self._vocab else 0.0

        # Calculate log probability (to avoid underflow)
        log_prob = 0.0
        for w1, w2 in zip(words[:-1], words[1:]):
            score = self.score_transition(w1, w2)
            if score == 0:
                return 0.0
            log_prob += math.log(score)

        # Convert back to probability
        return math.exp(log_prob)

    @property
    def vocabulary(self) -> set[str]:
        return self._vocab.copy()

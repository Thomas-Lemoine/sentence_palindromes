from src.language_model.base import LanguageModel


class UniformLanguageModel(LanguageModel):
    """Simple language model that assigns uniform probabilities"""

    def __init__(self):
        super().__init__()
        self._vocab: set[str] = set()

    def train(self, corpus: list[list[str]]) -> None:
        """Build vocabulary from corpus"""
        for sentence in corpus:
            for word in sentence:
                self._vocab.add(word.lower())

        # Update stats
        self.stats.unique_words = len(self._vocab)
        self.stats.total_words = sum(len(s) for s in corpus)
        self.stats.unique_bigrams = len(self._vocab) * len(self._vocab)
        self.stats.total_bigrams = sum(len(s) - 1 for s in corpus)
        self.stats.vocab_coverage = 1.0  # By definition for uniform model

    def score_transition(self, word1: str, word2: str) -> float:
        """Score transitions uniformly if both words in vocabulary"""
        word1, word2 = word1.lower(), word2.lower()
        if word1 in self._vocab and word2 in self._vocab:
            return 1.0
        return 0.0

    def score_sequence(self, words: list[str]) -> float:
        """Score sequence uniformly if all words in vocabulary"""
        words = [w.lower() for w in words]
        if all(w in self._vocab for w in words):
            return 1.0
        return 0.0

    @property
    def vocabulary(self) -> set[str]:
        return self._vocab.copy()

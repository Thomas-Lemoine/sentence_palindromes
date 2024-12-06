from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LanguageModelStats:
    """Statistics about the language model's data"""

    total_words: int = 0
    unique_words: int = 0
    total_bigrams: int = 0
    unique_bigrams: int = 0
    vocab_coverage: float = 0.0

    def __str__(self) -> str:
        return (
            f"Total Words: {self.total_words}\n"
            f"Unique Words: {self.unique_words}\n"
            f"Total Bigrams: {self.total_bigrams}\n"
            f"Unique Bigrams: {self.unique_bigrams}\n"
            f"Vocabulary Coverage: {self.vocab_coverage:.2%}"
        )


class LanguageModel(ABC):
    """Abstract base class for language models"""

    def __init__(self):
        self.stats = LanguageModelStats()

    @abstractmethod
    def train(self, corpus: list[list[str]]) -> None:
        """Train the model on a corpus of sentences

        Args:
            corpus: List of sentences, where each sentence is a list of words
        """
        pass

    @abstractmethod
    def score_transition(self, word1: str, word2: str) -> float:
        """Score the transition probability between two words

        Args:
            word1: First word
            word2: Second word

        Returns:
            Float score between 0 and 1, where higher is more likely
        """
        pass

    @abstractmethod
    def score_sequence(self, words: list[str]) -> float:
        """Score a complete sequence of words

        Args:
            words: Sequence of words to score

        Returns:
            Float score between 0 and 1, where higher is more likely
        """
        pass

    @property
    @abstractmethod
    def vocabulary(self) -> set[str]:
        """Get the model's vocabulary

        Returns:
            Set of words in the model's vocabulary
        """
        pass

import nltk
from nltk.corpus import brown


def get_vocabulary(top_n: int = 5000, min_word_len: int = 2) -> set[str]:
    """Get vocabulary from Brown corpus"""
    nltk.download("brown", quiet=True)
    freq_dist = nltk.FreqDist(
        word.lower() for word in brown.words() if word.isalpha() and len(word) >= min_word_len
    )
    return {word for word, _ in freq_dist.most_common(top_n)}

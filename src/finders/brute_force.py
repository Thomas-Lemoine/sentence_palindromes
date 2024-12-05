import itertools

import nltk
from nltk import FreqDist
from nltk.corpus import brown

# download required corpora
nltk.download("brown", quiet=True)

# configure parameters
TOP_N_WORDS = 5000
MIN_LENGTH = 4
MAX_LENGTH = 6

# generate a set of top n frequent words from the brown corpus
freq_list = FreqDist(
    word.lower()
    for word in brown.words()
    if word.isalpha() and ((word.islower() and len(word) > 1) or word == "I")
)
top_words = {word for word, _ in freq_list.most_common(TOP_N_WORDS)}


def is_palindrome(s: str) -> bool:
    return s == s[::-1]


# build a compatibility dictionary
def build_compatibility_dict(words):
    compatible = {}
    for w1 in words:
        rev_w1 = w1[::-1]
        compatible[w1] = [w2 for w2 in words if w2.endswith(rev_w1) or rev_w1.endswith(w2)]
    return compatible


compatible = build_compatibility_dict(top_words)


# generate palindrome sentences
def generate_palindromes(length):
    if length == 1:
        yield from (w for w in top_words if is_palindrome(w))
        return

    for w1, end_words in compatible.items():
        for w_last in end_words:
            if length == 2:
                combined = w1 + w_last
                if is_palindrome(combined):
                    yield f"{w1} {w_last}"
            else:
                middle_combinations = itertools.product(top_words, repeat=length - 2)
                for middle in middle_combinations:
                    combined = w1 + "".join(middle) + w_last
                    if is_palindrome(combined):
                        yield f"{w1} {' '.join(middle)} {w_last}"


# run the palindrome generator for lengths in the range
if __name__ == "__main__":
    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        for sentence in generate_palindromes(length):
            print(sentence)

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_word: bool = False
    word: str = ""


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word

    def find_words_with_prefix(self, prefix: str) -> Set[str]:
        """Find all words that start with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]

        words = set()

        def collect_words(node: TrieNode):
            if node.is_word:
                words.add(node.word)
            for child in node.children.values():
                collect_words(child)

        collect_words(node)
        return words

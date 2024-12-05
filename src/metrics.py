from dataclasses import dataclass, field


@dataclass
class PalindromeMetrics:
    generation_time: float = 0.0
    num_palindromes: int = 0
    avg_length: float = 0.0
    max_length: int = 0
    length_distribution: dict[int, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Generation Time: {self.generation_time:.2f}s\n"
            f"Number of Palindromes: {self.num_palindromes}\n"
            f"Average Length: {self.avg_length:.2f} words\n"
            f"Max Length: {self.max_length} words\n"
            f"Length Distribution: {dict(sorted(self.length_distribution.items()))}"
        )

# Sentence Palindromes

This project is our attempt at finding sentence palindromes, sentences that read the same way forwards and backwards.

## Usage

Run finders with uv:
```bash
uv run -m src.finders.[finder_name]
```

Available finders:
- llm

Example output:
```bash
...
Found: saga sill rachel notre plays lapse rose divide sores palsy alpert onleh carl lisa gas
Found: saga sill rachel notre plays lapse rose divide sores palsy alpert onleh carl lisa gas
Found: se saga sill rachel notre plays lapse rose divide sores palsy alpert onleh carl lisa gases
Found: step rachel notre plays lapse rose divide sores palsy alpert onleh carpets
Found: step rachel notre plays lapse rose divide sores palsy alpert onleh carpets
Found: oath turret rachel notre plays lapse rose divide sores palsy alpert onleh carter ruth tao
Found: oath turret rachel notre plays lapse rose divide sores palsy alpert onleh carter ruth tao
Found: flesh turret rachel notre plays lapse rose divide sores palsy alpert onleh carter ruth self
Found: flesh turret rachel notre plays lapse rose divide sores palsy alpert onleh carter ruth self
Found: nam regret rachel notre pl
...
```

## Finders

- [x] brute_force
- [x] basic
- [x] beam_search
- [x] middle_out
Edit: All of the above need to be translated into the new system.

## Notes

TODO: Remove/clean up notes

1. Middle-Out vs Side-In Construction
   - Middle-out construction can be very efficient because:
     - The middle character/position uniquely constrains both sides
     - You can build symmetrically outwards, reducing search space
     - It naturally handles both odd and even length palindromes
   - Your current side-in approach has to essentially guess both ends independently
   - Middle-out would let you:
     - Start with either a single character or empty center
     - Expand by adding matching characters/words to both sides
     - Prune invalid branches early

2. Trie-Based Optimization
   - Building a trie of your vocabulary could speed things up significantly
   - You could build two tries:
     - Forward trie: normal word storage
     - Reverse trie: storing reversed words
   - This would let you:
     - Quickly find words that could extend a palindrome
     - Efficiently check if a prefix/suffix has potential completions
     - Prune dead-end paths early in construction

3. Dynamic Programming Ideas
   - Classic DP for palindrome problems usually works on subproblems like:
     - Is substring[i:j] a palindrome?
     - What's the minimum number of cuts to make palindromes?
   - For sentence palindromes, we could try:
     - Storing valid palindromic word sequences of length k
     - Using them to build sequences of length k+1
     - Caching partial results to avoid recomputation
     - Using bit arrays to efficiently track valid combinations

4. Search Space Reduction
   - Pre-compute word compatibility classes:
     - Group words by their first/last n characters
     - Create palindrome-compatible word pairs
     - Build an adjacency graph of compatible words
   - Use prefix/suffix tables:
     - Index words by their prefixes and suffixes
     - Quick lookup for possible extensions
     - Eliminate impossible combinations early

5. Real-World Approaches
   - The most famous work is by Dr. William Tunstall-Pedoe who:
     - Used massive computer clusters
     - Generated billions of candidates
     - Applied natural language filtering
     - Found "Do geese see God?" and other classics
   - Academic approaches often focus on:
     - Constraint satisfaction problems
     - Genetic algorithms for optimization
     - Neural approaches for coherence

6. Alternative Algorithmic Approaches
   - Bidirectional search:
     - Build from both ends simultaneously
     - Meet in the middle
     - Can reduce search space dramatically
   - A* search with heuristics:
     - Estimate likelihood of completion
     - Prioritize promising partial solutions
   - Beam search:
     - Maintain k best partial solutions
     - Expand most promising candidates
     - Balance completeness vs efficiency

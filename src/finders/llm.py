from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertForMaskedLM,
    T5ForConditionalGeneration,
)

from src.finders.base import PalindromeFinder
from src.finders.scorers import (
    CharacterRepetitionScorer,
    PartOfSpeechBalanceScorer,
    RepetitionPenaltyScorer,
    T5TransformerScorer,
    TransformerScorer,
    WordDiscriminator,
)

if __name__ == "__main__":
    from src.utils import get_vocabulary

    # bigscience/bloomz-560m, google/flan-t5-small, prajjwal1/bert-tiny, gpt2
    model_name = "gpt2"
    if "t5" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformer_scorer = T5TransformerScorer(
            model, AutoTokenizer.from_pretrained(model_name)
        )
    elif "bert" in model_name:
        model = BertForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformer_scorer = TransformerScorer(model, tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformer_scorer = TransformerScorer(model, tokenizer)

    scorers = [
        transformer_scorer,
        CharacterRepetitionScorer(),
        RepetitionPenaltyScorer(),
        # WordLengthScorer(),
        PartOfSpeechBalanceScorer(),
        WordDiscriminator(
            words=["m", "t", "o", "c", "n", "e", "p", "v", "h", "af", "la"]
        ),
    ]

    vocabulary = get_vocabulary(top_n=8_000, min_word_len=1)
    finder = PalindromeFinder(
        vocabulary=vocabulary,
        scorers=scorers,
        branching_factor=34,
    )

    for palindrome in finder.grow_palindromes(["be"], center_pos=0, depth=8):
        print(f" Found: {palindrome}")

    for palindrome in finder.generate_palindromes(depth=8):
        print(f" Found: {palindrome}")

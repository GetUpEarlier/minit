from abc import abstractmethod
from typing import List


class Tokenizer:
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        ...

    @abstractmethod
    def detokenize(self, ids: List[int]) -> str:
        ...


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        super().__init__()
        import sentencepiece
        self.sentence_piece = sentencepiece.SentencePieceProcessor(path)

    def tokenize(self, text: str) -> List[int]:
        [input_ids,] = self.sentence_piece.tokenize([
            text
        ])
        return input_ids

    def detokenize(self, ids: List[int]) -> str:
        text = ""
        for id in ids:
            text += self.sentence_piece.id_to_piece(id)
        return text


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        super().__init__()
        import transformers
        self.hugging_face = transformers.PreTrainedTokenizerFast(tokenizer_file=path)

    def tokenize(self, text: str) -> List[int]:
        input_ids = self.hugging_face.encode(text, add_special_tokens=False)
        return input_ids

    def detokenize(self, ids: List[int]) -> str:
        text = ""
        for id in ids:
            text += self.hugging_face.decode(id)
        return text

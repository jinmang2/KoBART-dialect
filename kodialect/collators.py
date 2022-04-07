import torch
from typing import List, Dict


class TextCNNCollator:
    """ TextCNN Data Collator """

    def __init__(
        self, tokenizer, max_length=510, text_col_name: str = "text"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col_name = text_col_name

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            text=[x[self.text_col_name] for x in batch],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )
        labels = torch.LongTensor([x["label"] for x in batch])
        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels,
        }


class KoDialectCollator:
    """ Korean Dialect Data Collator """

    def __init__(self, tokenizer, max_length=510):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            text=[x["source"] for x in batch],
            text_pair=[x["target"] for x in batch],
            src_langs=[x["src_lang"] for x in batch],
            tgt_langs=[x["tgt_lang"] for x in batch],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )
        return tokenized
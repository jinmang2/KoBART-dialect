import torch


class TextCNNCollator:
    def __init__(self, tokenizer, max_length=510):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = self.tokenizer(
            text=[x["sentence"] for x in batch],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )["input_ids"]
        labels = torch.LongTensor([x["label"] for x in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class KoDialectCollator:
    def __init__(self, tokenizer, max_length=510):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        return self.tokenizer(
            text=[x["source"] for x in batch],
            text_pair=[x["target"] for x in batch],
            src_langs=[x["src_lang"] for x in batch],
            tgt_langs=[x["tgt_lang"] for x in batch],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )

import torch
from typing import List, Optional, Dict, Union
from transformers import PreTrainedTokenizerFast
from contextlib import contextmanager
from tokenizers.processors import TemplateProcessing


class KodialectTokenizerFast(PreTrainedTokenizerFast):

    code_to_langs = {
        '표준': '[kr_KR]',
        '충청': '[kr_CC]',
        '강원': '[kr_GW]',
        '경상': '[kr_GS]',
        '제주': '[kr_JJ]',
        '전라': '[kr_JL]',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_code_to_id = {
            code: self.vocab.get(code, self.unk_token_id)
            for code in self.code_to_langs.values()
        }

    @contextmanager
    def as_target_tokenizer(self):
        self.backend_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[("<s>", 0), ("</s>", 1)],
        )
        yield self
        self.backend_tokenizer.post_processor = TemplateProcessing(
            single="$A",
            pair="$A $B:1",
        )

    def __call__(
        self,
        text: List[str],
        text_pair: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        max_length: int = 512,
        max_target_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        **kwargs,
    ):
        if src_langs is None or tgt_langs is None:
            return super().__call__(
                text=text,
                text_pair=text_pair,
                padding=padding,
                max_length=max_length,
                **kwargs,
            )

        assert text_pair is not None, "seq2seq batch requires text_pair"

        return self.prepare_seq2seq_batch(
            src_texts=text,
            src_langs=src_langs,
            tgt_texts=text_pair,
            tgt_langs=tgt_langs,
            max_length=max_length,
            max_target_length=max_target_length,
            padding=padding,
            **kwargs,
        )

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_langs: List[str],
        tgt_texts: List[str],
        tgt_langs: List[str],
        return_tensors: str = 'pt',
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        **kwargs,
    ):
        if isinstance(src_texts, str):
            src_texts = [src_texts]

        if isinstance(src_langs, str):
            src_langs = [src_langs] * len(src_texts)

        if isinstance(tgt_texts, str):
            tgt_texts = [tgt_texts]

        if isinstance(tgt_langs, str):
            tgt_langs = [tgt_langs] * len(tgt_texts)

        src_langs = [self.code_to_langs.get(lang, lang) for lang in src_langs]
        tgt_langs = [self.code_to_langs.get(lang, lang) for lang in tgt_langs]

        if max_length is None:
            max_length = self.model_max_length

        model_inputs = super().__call__(
            src_texts,
            return_tensors="pt",
            padding=padding,
            add_special_tokens=False,
            max_length=max_length - 2,
            return_token_type_ids=False,
            return_attention_mask=True,
            **kwargs,
        )
        model_inputs = self.add_language_tokens(model_inputs, src_langs, tgt_langs)

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length

        with self.as_target_tokenizer():
            labels = super().__call__(
                tgt_texts,
                return_tensors="pt",
                padding=padding,
                add_special_tokens=True,
                max_length=max_length - 2,
                return_token_type_ids=False,
                return_attention_mask=True,
                **kwargs,
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def add_language_tokens(self, tokens, src_langs, tgt_langs):
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_added_ids, token_added_masks = [], []

        for input_id, atn_mask, src_lang, tgt_lang in zip(
            input_ids, attention_mask, src_langs, tgt_langs
        ):
            maximum_idx = [i for i, val in enumerate(input_id) if val != self.pad_token_id]

            if len(maximum_idx) == 0:
                idx_to_add = 0
            else:
                idx_to_add = (max(maximum_idx) + 1)

            src_lang = self.lang_code_to_id[src_lang]
            tgt_lang = self.lang_code_to_id[tgt_lang]
            eos = self.eos_token_id

            input_id = torch.cat(
                [
                    torch.tensor([src_lang], requires_grad=False),
                    input_id[:idx_to_add],
                    torch.tensor([tgt_lang, eos], requires_grad=False),
                    input_id[idx_to_add:]
                ]
            ).long()

            atn_mask = torch.cat(
                [
                    torch.tensor([1], requires_grad=False),
                    atn_mask[:idx_to_add],
                    torch.tensor([1, 1], requires_grad=False),
                    atn_mask[idx_to_add:],
                ]
            ).long()

            token_added_ids.append(input_id.unsqueeze(0))
            token_added_masks.append(atn_mask.unsqueeze(0))

        tokens["input_ids"] = torch.cat(token_added_ids, dim=0)
        tokens["attention_mask"] = torch.cat(token_added_masks, dim=0)
        return tokens

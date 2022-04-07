import os
import json
import shutil
from zipfile import ZipFile
from .tokenization_kodialect import (
    PreTrainedTokenizerFast,
    KodialectTokenizerFast
)
from ..kobart_utils import download


def get_kobart_tokenizer(name: str, save_path: str):
    if name == "style_classification":
        tokenizer_class = PreTrainedTokenizerFast
    elif name == "style_transfer":
        tokenizer_class = KodialectTokenizerFast
    else:
        raise ValueError

    file_path, is_cached = download(
        url="s3://skt-lsl-nlp-model/KoBART/tokenizers/kobart_base_tokenizer_cased_cf74400bce.zip",
        chksum="cf74400bce",
        cachedir=save_path,
    )
    cachedir_full = os.path.expanduser(save_path)
    if (
        not os.path.exists(os.path.join(cachedir_full, "emji_tokenizer"))
        or not is_cached
    ):
        if not is_cached:
            shutil.rmtree(
                os.path.join(cachedir_full, "emji_tokenizer"), ignore_errors=True
            )
        zipf = ZipFile(os.path.expanduser(file_path))
        zipf.extractall(path=cachedir_full)
    tok_path = os.path.join(cachedir_full, "emji_tokenizer/model.json")

    if not is_cached:
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_model = json.load(f)

        special_tokens = [
            "[kr_KR]", "[kr_CC]", "[kr_GW]", "[kr_GS]", "[kr_JJ]", "[kr_JL]",
            "[NAME]", "[ADDRESS]", "[OTHER]"
        ]
        for idx, special_token in zip(range(7, 7 + len(special_tokens)), special_tokens):
            tok_model["added_tokens"][idx]["content"] = special_token
            value = tok_model["model"]["vocab"].pop(f"<unused{idx-7}>")
            tok_model["model"]["vocab"][special_token] = value

        tok_model["post_processor"] = {
            "type": "RobertaProcessing",
            "sep": ["</s>", 1],
            "cls": ["<s>", 0],
            "trim_offsets": True,
            "add_prefix_space": True,
        }

        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(tok_model, f, ensure_ascii=False)

    kobart_tokenizer = tokenizer_class(
        tokenizer_file=tok_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    return kobart_tokenizer
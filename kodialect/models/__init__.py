import os
import shutil
from zipfile import ZipFile

from .bart import BartConfig, BartForConditionalGeneration
from .textcnn import TextCNNConfig, TextCNNForSequenceClassification
from ..kobart_utils import download


def get_kobart_model_path(save_path: str):
    model_zip, is_cached = download(
        url="s3://skt-lsl-nlp-model/KoBART/models/kobart_base_cased_ff4bda5738.zip",
        chksum="ff4bda5738",
        cachedir=save_path,
    )
    cachedir_full = os.path.join(os.getcwd(), save_path)
    model_path = os.path.join(cachedir_full, "kobart_from_pretrained")
    if not os.path.exists(model_path) or not is_cached:
        if not is_cached:
            shutil.rmtree(model_path, ignore_errors=True)
        zipf = ZipFile(os.path.expanduser(model_zip))
        zipf.extractall(path=cachedir_full)
    return model_path
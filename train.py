import os
import sys
from functools import partial
from datasets import load_from_disk
from transformers.hf_argparser import HfArgumentParser

from kodialect.tokenizers import get_kobart_tokenizer
from kodialect.collators import TextCNNCollator, KoDialectCollator
from kodialect.metrics import (
    compute_sc_metrics,
    compute_st_metrics,
)
from kodialect.models import (
    get_kobart_model_path,
    TextCNNConfig,
    TextCNNForSequenceClassification,
    BartConfig,
    BartForConditionalGeneration,
)
from kodialect.trainers import (
    StyleClassifierTrainer,
    KodialectTrainer,
    StyleClassifierTrainingArguments,
    KodialectTrainingArguments,
)


os.environ["WANDB_PROJECT"] = "kobart-dialect"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"


class StyleClassification:
    name = "style_classification"
    data_path = "data/style_classification"
    save_path = "jinmang2/textcnn-ko-dialect-classifier"
    args = StyleClassifierTrainingArguments
    trainer = StyleClassifierTrainer
    config = TextCNNConfig
    model = TextCNNForSequenceClassification
    metric = compute_sc_metrics
    collator = TextCNNCollator


class StyleTransfer:
    name = "style_transfer"
    data_path = "data/style_transfer"
    save_path = "jinmang2/kobart-dialect"
    args = KodialectTrainingArguments
    trainer = KodialectTrainer
    config = BartConfig
    model = BartForConditionalGeneration
    metric = compute_st_metrics
    collator = KoDialectCollator


if __name__ == "__main__":
    train_mode = sys.argv.pop(1)
    if train_mode in ["sc", "style_classifier", "style_classification"]:
        module = StyleClassification
    elif train_mode in ["st", "style_transfer"]:
        module = StyleTransfer
    else:
        raise ArgumentError
    parser = HfArgumentParser(module.args)
    training_args, = parser.parse_args_into_dataclasses()
    
    # get kobart tokenizer
    tokenizer = get_kobart_tokenizer(module.name, module.save_path)

    config = None
    if module.name == "style_classification":
        config = module.config(vocab_size=len(tokenizer))
        model = module.model(config)
    else:
        model_path = get_kobart_model_path(module.save_path)
        model = module.model.from_pretrained(model_path)

    metric_fn = module.metric

    data_collator = module.collator(tokenizer)
    dataset = load_from_disk(module.data_path)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].select(range(10000)),
        eval_dataset=dataset["valid"].select(range(1000)),
        data_collator=data_collator,
        compute_metrics=metric_fn,
    )
    if module.name == "style_transfer":
        classifier = StyleClassification.model.from_pretrained(
            StyleClassification.save_path
        )
        trainer_kwargs.update({"classifier": classifier, "tokenizer": tokenizer})

    trainer = module.trainer(**trainer_kwargs)
    trainer.train()

    trainer.args.output_dir = module.save_path
    trainer.save_state()
    trainer.save_model()
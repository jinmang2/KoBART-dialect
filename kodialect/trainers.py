from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from transformers import (
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_pt_utils import nested_detach


@dataclass
class StyleClassifierTrainingArguments(TrainingArguments):
    pass


@dataclass
class KodialectTrainingArguments(Seq2SeqTrainingArguments):
    temperature: float = field(default=1.0)


class StyleClassifierTrainer(Trainer):
    pass


class KodialectTrainer(Seq2SeqTrainer):

    def __init__(self, classifier: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokenizer, "Requiers tokenizer"
        self.classifier = classifier
        self.classifier.to(self.args.device)
        self.classifier.eval()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False
    ):
        outputs = model(**inputs)
        # Cross Entropy Loss
        loss_ce = outputs.loss
        if "labels" in inputs:
            labels = inputs.pop("labels")
        idx = labels.ne(self.tokenizer.pad_token_id).sum(dim=-1)
        # Style Classifier based reward Loss
        loss_sc = self.cal_sc_loss(outputs.logits, inputs, idx)
        # BLEU based reward loss
        loss_co = self.cal_bl_loss(outputs.logits, labels, idx)
        loss = loss_ce + loss_sc + loss_co
        return (loss, outputs) if return_outputs else loss

    def sample_with_temperature(self, probs, temperature=1.0):
        # sample
        sample_idx = torch.zeros(probs.size(0), probs.size(1)).to(self.args.device)
        sample_probs = torch.zeros(probs.size(0), probs.size(1)).to(self.args.device)
        if temperature < 1.0:
            temp = torch.exp(torch.div(torch.log(probs + 1e-20), temperature))
        else:
            temp = probs
        for i, s in enumerate(temp):
            temp_idx = torch.multinomial(s, 1) # shape = (seq_len, 1)
            temp_probs = s.gather(1, temp_idx) # shape = (seq_len, 1)
            sample_idx[i] = temp_idx.squeeze(1)
            sample_probs[i] = temp_probs.squeeze(1)

        return sample_probs, sample_idx.long()

    def cal_sc_loss(self, logits, inputs, idx):
        out = torch.softmax(logits, dim=-1)
        sample_probs, sample_idx = self.sample_with_temperature(out, self.args.temperature)

        tgt = []
        for i, s in zip(idx.cpu(), sample_idx):
            e = torch.arange(len(s))[s.eq(self.tokenizer.eos_token_id)]
            e = e[0] if 0 < len(e) and 4 < e[0] < i else i-1
            tgt.append(s[:e].cpu().tolist())
        tgt = self.tokenizer.pad({"input_ids":tgt}, return_tensors="pt")
        tgt_idx = tgt["input_ids"].to(self.args.device)
        tgt_cls = torch.softmax(self.classifier(tgt_idx).logits.detach(), dim=-1)

        tgt_reward = tgt_cls[:, 0] - tgt_cls[:, 1]

        input_ids = inputs["input_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        idxs = torch.arange(0, input_ids.size(0))
        lang_idxs = input_ids.eq(eos_token_id).long().argmax(dim=-1) - 1
        standard_token = self.tokenizer.lang_code_to_id["[kr_KR]"]
        labels = input_ids[idxs, lang_idxs].ne(standard_token).long()

        tgt_reward[labels] = tgt_reward[labels] * -1

        return self.cal_reward_loss(sample_probs, tgt_reward, idx)

    def cal_bl_reward(self, inp, tgt):
        smooth = SmoothingFunction()
        blues = []
        for hyp, ref in zip(inp, tgt):
            blues.append(
                sentence_bleu([ref], hyp, smoothing_function=smooth.method1)
            )
        blues = torch.FloatTensor(blues).to(self.args.device)

        return blues

    def cal_bl_loss(self, logits, labels, idx):
        out = torch.softmax(logits, dim=-1)
        sample_probs, sample_idx = self.sample_with_temperature(out, self.args.temperature)
        greedy_probs, greedy_idx = torch.max(out, dim=-1)

        tgt_sam, tgt_gre, tgt_ref = [], [], []
        for i, s, g, t in zip(idx.cpu(), sample_idx, greedy_idx, labels):
            s_e = torch.arange(len(s))[s.eq(self.tokenizer.eos_token_id)]
            s_e = s_e[0] if 0 < len(s_e) and 0 < s_e[0] < i else i-1
            g_e = torch.arange(len(g))[g.eq(self.tokenizer.eos_token_id)]
            g_e = g_e[0] if 0 < len(g_e) and 0 < g_e[0] < i else i-1

            tgt_sam.append(s[:s_e].cpu().tolist())
            tgt_gre.append(g[:g_e].cpu().tolist())
            tgt_ref.append(t[1:i].cpu().tolist())

        tgt_sam = self.cal_bl_reward(tgt_sam, tgt_ref)
        tgt_gre = self.cal_bl_reward(tgt_gre, tgt_ref)
        loss_co = self.cal_reward_loss(sample_probs, (tgt_gre-tgt_sam)*0.2, idx)

        return loss_co

    def cal_reward_loss(self, sample_probs, reward, idx=None):
        sample_probs = sample_probs.contiguous()
        sample_logprobs = torch.log(sample_probs)
        reward = reward.unsqueeze(1).contiguous()
        if idx is not None:
            batch_size, max_len = sample_probs.size()
            mask = torch.zeros(batch_size, max_len).to(self.args.device)
            for i, ix in enumerate(idx):
                mask[i, :ix] = 1
            mask = mask.float().contiguous()
            output = -sample_logprobs * reward * mask
            output = (output.sum(dim=-1) / mask.sum(dim=-1)).mean()
        else:
            output = -sample_logprobs * reward
            output = output.mean()

        return output

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        if self.args.predict_with_generate:
            super(Seq2SeqTrainer).prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs

        if self.args.prediction_loss_only:
            return (loss, None, None)

        input_ids = inputs["input_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        idxs = torch.arange(0, input_ids.size(0))
        lang_idxs = input_ids.eq(eos_token_id).long().argmax(dim=-1) - 1
        standard_token = self.tokenizer.lang_code_to_id["[kr_KR]"]
        labels = input_ids[idxs, lang_idxs].ne(standard_token).long()

        logits = nested_detach(logits)[0]
        preds = logits.argmax(-1)
        tgt = []
        for pred in preds:
            e = torch.arange(len(pred))[pred.eq(self.tokenizer.eos_token_id)]
            e = e[0] if 0 < len(e) and e[0] < 30 else 30
            tgt.append(pred[:e].cpu().tolist())
        tgt = self.tokenizer.pad({"input_ids": tgt}, return_tensors="pt")
        tgt = tgt["input_ids"].to(self.args.device)
        with torch.no_grad():
            logits = self.classifier(tgt).logits # y_hat

        return (loss, logits, labels)

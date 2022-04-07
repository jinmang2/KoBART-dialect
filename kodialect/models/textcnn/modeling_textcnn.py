import torch
import torch.nn as nn
from typing import List
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput
from .configuration_textcnn import TextCNNConfig


@dataclass
class TextCNNModelOutput(ModelOutput):
    last_hidden_states: torch.FloatTensor = None
    ngram_feature_maps: List[torch.FloatTensor] = None


@dataclass
class TextCNNSequenceClassificerOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    ngram_feature_maps: List[torch.FloatTensor] = None


class TextCNNPreTrainedModel(PreTrainedModel):
    config_class = TextCNNConfig
    base_model_prefix = "textcnn"

    def _init_weights(self, module):
        return NotImplementedError

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor(
            [[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device
        )
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        dummy_inputs


class TextCNNModel(TextCNNPreTrainedModel):
    """ A Style classifier Text-CNN """

    def __init__(self, config):
        super().__init__(config)
        self.embeder = nn.Embedding(config.vocab_size, config.embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, config.embed_dim))
            for (n, f) in zip(config.num_filters, config.filter_sizes)
        ])

    def get_input_embeddings(self):
        return self.embeder

    def set_input_embeddings(self, value):
        self.embeder = value

    def forward(self, input_ids):
        # input_ids.shape == (bsz, seq_len)
        x = self.embeder(input_ids).unsqueeze(1) # add channel dim
        # x.shape == (bsz, 1, seq_len, emb_dim)
        convs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        # convs[i].shape == (bsz, n_filter[i], ngram_seq_len)
        pools = [torch.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        # pools[i].shape == (bsz, n_filter[i])
        outputs = torch.cat(pools, 1)
        # outputs.shape == (bsz, feature_dim)

        return TextCNNModelOutput(
            last_hidden_states=outputs,
            ngram_feature_maps=pools,
        )


class TextCNNForSequenceClassification(TextCNNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.feature_dim = sum(config.num_filters)
        self.textcnn = TextCNNModel(config)
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), config.num_labels)
        )

    def forward(self, input_ids, labels=None):
        # input_ids.shape == (bsz, seq_len)
        # labels.shape == (bsz,)
        outputs = self.textcnn(input_ids)
        # outputs.shape == (bsz, feature_dim)
        logits = self.fc(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return TextCNNSequenceClassificerOutput(
            loss=loss,
            logits=logits,
            last_hidden_states=outputs.last_hidden_states,
            ngram_feature_maps=outputs.ngram_feature_maps,
        )

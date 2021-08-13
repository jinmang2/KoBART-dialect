from transformers.configuration_utils import PretrainedConfig


class TextCNNConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=30000,
        embed_dim=300,
        filter_sizes=[1,2,3,4,5],
        num_filters=[128]*5,
        dropout=0.5,
        num_labels=2,
        id2label={0:"standard", 1:"dialect"},
        label2id={"standard":0, "dialect":1},
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=3,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout = dropout

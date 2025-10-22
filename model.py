import torch
import torch.nn as nn
from transformers import AutoModel
from functools import partial


class AttentionPooling(nn.Module):
    """Single-head additive attention pooling over token embeddings."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, L, H]
        scores = self.W2(torch.tanh(self.W1(hidden_states))).squeeze(-1)  # [B, L]
        if attention_mask is not None:
            mask = attention_mask.to(torch.bool)
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)                            # [B, L]
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)   # [B, H]
        return pooled


class CrossEncoderScorer(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        load_in_4bit: bool = False,
        pool_type: str = "attention",
    ):
        """
        pool_type: 'attention' (recommended), 'cls', or 'mean'
        """
        super().__init__()

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                else torch.float16
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.backbone = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                add_pooling_layer=False,   # avoid pooler warning
                device_map="auto",
            )
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                add_pooling_layer=False,   # avoid pooler warning
            )

        hidden_size = self.backbone.config.hidden_size
        self.pool_type = pool_type.lower()

        if self.pool_type == "attention":
            self.pool = AttentionPooling(hidden_size)
        elif self.pool_type in ("mean", "cls"):
            self.pool = None
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")

        self.dropout = nn.Dropout(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        self.head = nn.Linear(hidden_size, 1)

    def _pool(self, last_hidden_state, attention_mask):
        # last_hidden_state: [B, L, H]
        if self.pool_type == "attention":
            return self.pool(last_hidden_state, attention_mask)  # [B, H]
        elif self.pool_type == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
            summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
            return summed / denom
        else:  # 'cls'
            return last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self._pool(out.last_hidden_state, attention_mask)  # [B, H]
        pooled = self.dropout(pooled)
        logits = self.head(pooled).squeeze(-1)  # [B]
        return logits


def get_model_factory(arch_key: str, *, model_name: str = "roberta-base", load_in_4bit: bool = False):
    """
    Returns a factory (callable) that instantiates CrossEncoderScorer with the desired pooling.
    Usage:
        Factory = get_model_factory("cross_encoder_attention", model_name="roberta-base")
        model = Factory()   # creates the model when you need it
    """
    options = {
        "cross_encoder": partial(CrossEncoderScorer, model_name=model_name, load_in_4bit=load_in_4bit, pool_type="cls"),
        "cross_encoder_mean_pooling": partial(CrossEncoderScorer, model_name=model_name, load_in_4bit=load_in_4bit, pool_type="mean"),
        "cross_encoder_attention": partial(CrossEncoderScorer, model_name=model_name, load_in_4bit=load_in_4bit, pool_type="attention"),
    }
    if arch_key not in options:
        raise ValueError(f"Unknown model key '{arch_key}'. Available: {list(options.keys())}")
    return options[arch_key]

def  get_model(model_name:str):
    models = {
        "cross_encoder": CrossEncoderScorer(pool_type="cls"),
        "cross_encoder_mean_pooling": CrossEncoderScorer(pool_type="mean"),
        "cross_encoder_attention": CrossEncoderScorer(pool_type="attention"),
        }
    return models[model_name]
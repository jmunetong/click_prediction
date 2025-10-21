import torch
import torch.nn as nn
from transformers import AutoModel
from functools.partial import partial


class CrossEncoderScorer(nn.Module):
    def __init__(self, model_name: str, load_in_4bit: bool = False):
        super().__init__()
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.backbone = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from the model config
        hidden_size = self.backbone.config.hidden_size
        
        # Scoring head: maps CLS representation to a single score
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B*N, L] tensor of token ids
            attention_mask: [B*N, L] tensor of attention mask
            
        Returns:
            logits: [B*N] tensor of scores
        """
        # Encode the input
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
       
        # Use [CLS] token representation (first token)
        cls = out.last_hidden_state[:, 0, :]  # [B*N, H]
        
        # Score each pair
        logits = self.head(cls).squeeze(-1)   # [B*N]
        
        return logits

class AttentionPooling(nn.Module):
    """
    Single-head additive attention pooling over token embeddings.
    Given hidden states H: [B, L, H], returns pooled: [B, H].
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, L, H]
        # attention_mask: [B, L] with 1 for tokens to keep, 0 for pad
        # scores: [B, L]
        scores = self.W2(torch.tanh(self.W1(hidden_states))).squeeze(-1)

        if attention_mask is not None:
            # ensure boolean mask; mask out pads with large negative
            mask = attention_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=-1)                # [B, L]
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # [B, H]
        return pooled


class CrossEncoderScorer(nn.Module):
    def __init__(self, model_name: str, load_in_4bit: bool = False,
                 pool_type: str = "attention"):
        """
        pool_type: 'attention' (recommended), 'cls', or 'mean'
        """
        super().__init__()

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            # For Ada/Ampere, bfloat16 compute is generally nicer than fp16.
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.backbone = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.hidden_size
        self.pool_type = pool_type.lower()

        if self.pool_type == "attention":
            self.pool = AttentionPooling(hidden_size)
        elif self.pool_type == "mean":
            self.pool = None  # mean pooling doesn't need params
        elif self.pool_type == "cls":
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
            # mask-aware mean pooling
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
            summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
            return summed / denom
        else:  # 'cls'
            return last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [B*N, L]
        attention_mask: [B*N, L]
        returns logits: [B*N]
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self._pool(out.last_hidden_state, attention_mask)  # [B*N, H]
        pooled = self.dropout(pooled)
        logits = self.head(pooled).squeeze(-1)  # [B*N]
        return logits


def  get_model(model_name:str):
    models = {
        "cross_encoder": partial(CrossEncoderScorer, pool_type="cls"),
        "cross_encoder_mean_pooling": partial(CrossEncoderScorer, pool_type="mean"),
        "cross_encoder_attention":  partial(CrossEncoderScorer, pool_type="attention"),
        }
    return models[model_name]
import torch.nn as nn
from transformers import AutoModel

class CrossEncoderScorer(nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 1)   # 1 logit per pair

    def forward(self, input_ids, attention_mask):
        # input_ids / attention_mask: [B*N, L]
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use the first token's hidden state (RoBERTa CLS token at position 0)
        cls = out.last_hidden_state[:, 0, :]          # [B*N, H]
        logits = self.head(cls).squeeze(-1)           # [B*N]
        return logits
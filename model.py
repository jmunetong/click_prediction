import torch
import torch.nn as nn
from transformers import AutoModel

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
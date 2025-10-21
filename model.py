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
            self.encoder = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
        
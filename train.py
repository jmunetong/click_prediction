from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

from dataset import ParquetDataset, collate_queries
from model import CrossEncoderScorer

# Build dataset/data loader
# If single file: paths = ["tokenized_dataset.parquet"]
paths = ["preprocessed_click_train.parquet"]
dataset = ParquetDataset(paths)
loader = DataLoader(
    dataset,
    batch_size=8,                 # batches of *queries*
    num_workers=4,
    collate_fn=collate_queries,
    pin_memory=True,
    persistent_workers=True,
)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU) device")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MPS not available — using CPU")
print("About to load model to device")
model = CrossEncoderScorer("roberta-base").to(device)
print("Model has been loaded to device")
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
print("Starting for loop loading with batch sizes")
for step, batch in enumerate(loader):
    print(f"Processing step {step}...")
    input_ids = batch["input_ids"].to(device)          # [B, N, L]
    attention = batch["attention_mask"].to(device)     # [B, N, L]
    cand_mask = batch["candidate_mask"].to(device)     # [B, N]
    labels = batch["label_index"].to(device)           # [B]

    B, N, L = input_ids.shape
    print("Flattening inputs...")
    # Flatten pairs → run model ONCE
    input_ids_flat = input_ids.reshape(B * N, L)
    attention_flat = attention.reshape(B * N, L)
    print("Running model...")
    logits_flat = model(input_ids_flat, attention_flat)      # [B*N]
    print("Reshaping logits...")
    # Reshape logits to [B, N]
    logits = logits_flat.view(B, N)

    # Mask padded candidates: set logits to -inf so softmax gives them 0 prob
    logits = logits.masked_fill(~cand_mask, float("-inf"))
    print("Computing loss...")
    # Cross-entropy over N (dim=1). This is per-query softmax.
    loss = F.cross_entropy(logits, labels)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if step % 50 == 0:
        with torch.no_grad():
            # top-1 accuracy (ignore padded slots by construction)
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
        print(f"step {step} | loss {loss.item():.4f} | acc {acc:.3f}")
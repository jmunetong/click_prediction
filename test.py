from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import os
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
import math

from dataset import ParquetDataset, collate_queries
from model import CrossEncoderScorer

# Training configuration
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01  # L2 regularization
WARMUP_STEPS = 500   # Linear warmup steps
MAX_GRAD_NORM = 1.0  # Gradient clipping
LOG_INTERVAL = 50
EVAL_INTERVAL = 500  # Evaluate on validation set every N steps (optional)

# Create checkpoint directory
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Build dataset/data loader
train_paths = ["preprocessed_click_train.parquet"]
train_dataset = ParquetDataset(train_paths)


# Build test dataset/loader
test_paths = ["preprocessed_click_test.parquet"]
test_dataset = ParquetDataset(test_paths)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    collate_fn=collate_queries,
    pin_memory=True,
    persistent_workers=True,
)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU) device")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

model = CrossEncoderScorer("roberta-base").to(device)

# Load best model
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']}, step {checkpoint['global_step']} (train acc: {checkpoint['accuracy']:.3f})")

model.eval()

total_correct = 0
total_samples = 0
total_loss = 0.0
num_batches = 0

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        cand_mask = batch["candidate_mask"].to(device)
        labels = batch["label_index"].to(device)

        B, N, L = input_ids.shape
        input_ids_flat = input_ids.reshape(B * N, L)
        attention_flat = attention.reshape(B * N, L)

        logits_flat = model(input_ids_flat, attention_flat)
        logits = logits_flat.view(B, N)
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=1)
        
        total_correct += (pred == labels).sum().item()
        total_samples += B
        total_loss += loss.item()
        num_batches += 1

        if step % LOG_INTERVAL == 0:
            print(f"test step {step} | loss {loss.item():.4f}")

# Calculate final test metrics
test_acc = total_correct / total_samples
test_loss = total_loss / num_batches

print("\n" + "="*50)
print("TESTING RESULTS")
print("="*50)
print(f"Test Accuracy: {test_acc:.4f} ({total_correct}/{total_samples})")
print(f"Test Loss: {test_loss:.4f}")
print(f"Total test batches: {num_batches}")
print(f"Total test samples: {total_samples}")
print("="*50)
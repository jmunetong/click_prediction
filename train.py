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

# Get total samples for accurate scheduler calculation
total_samples = train_dataset.get_total_samples()
print(f"Total training samples: {total_samples}")
estimated_steps_per_epoch = total_samples // BATCH_SIZE
total_steps = NUM_EPOCHS * estimated_steps_per_epoch
print(f"Estimated steps per epoch: {estimated_steps_per_epoch}")
print(f"Total training steps: {total_steps}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    collate_fn=collate_queries,
    pin_memory=True,
    persistent_workers=True,
)

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

# Optimizer with weight decay (AdamW includes L2 regularization)
# Separate learning rates for encoder vs. task head (optional but recommended)
encoder_params = []
head_params = []
for name, param in model.named_parameters():
    if 'encoder' in name:
        encoder_params.append(param)
    else:
        head_params.append(param)

optim = torch.optim.AdamW([
    {'params': encoder_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
    {'params': head_params, 'lr': LEARNING_RATE * 5, 'weight_decay': 0.0}  # Higher LR for head
], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler with warmup - using LambdaLR
def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        # Linear warmup
        return float(current_step) / float(max(1, WARMUP_STEPS))
    else:
        # Cosine decay
        progress = float(current_step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optim, lr_lambda)

# Tracking best model
best_acc = 0.0
best_model_path = checkpoint_dir / "best_model.pt"
last_checkpoint_path = checkpoint_dir / "last_model.pt"

# For tracking training metrics
train_metrics = {
    'losses': [],
    'accuracies': [],
    'learning_rates': []
}

print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Warmup Steps: {WARMUP_STEPS}")
print(f"Gradient Clipping: {MAX_GRAD_NORM}")
print("="*50)

global_step = 0
training_complete = False

for epoch in range(NUM_EPOCHS):
    if training_complete:
        break
        
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 50)
    
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Check if we've reached total_steps
        if global_step >= total_steps:
            print(f"\nReached total_steps ({total_steps}). Stopping training.")
            training_complete = True
            break
            
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

        optim.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        optim.step()
        scheduler.step()  # Update learning rate
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        # Track metrics
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == labels).sum().item()
            epoch_correct += correct
            epoch_total += B
            epoch_loss += loss.item()
            batch_count += 1
            
            # Store metrics
            train_metrics['losses'].append(loss.item())
            train_metrics['accuracies'].append(correct / B)
            train_metrics['learning_rates'].append(current_lr)

        if global_step % LOG_INTERVAL == 0:
            batch_acc = correct / B
            print(f"epoch {epoch + 1} | step {global_step}/{total_steps} | batch {batch_idx} | "
                  f"loss {loss.item():.4f} | acc {batch_acc:.3f} | lr {current_lr:.2e}")
            
            # Save best model based on batch accuracy
            if batch_acc > best_acc:
                best_acc = batch_acc
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': batch_acc,
                    'loss': loss.item(),
                    'train_metrics': train_metrics,
                }, best_model_path)
                print(f"  → New best model saved! acc={batch_acc:.3f}")
        
        global_step += 1
    
    # Save checkpoint at end of each epoch (only if not already complete)
    if not training_complete:
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
        }, last_checkpoint_path)
    
    # End of epoch summary
    if batch_count > 0:
        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_acc = epoch_correct / epoch_total
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Batches processed: {batch_count}")
        print(f"  Samples processed: {epoch_total}")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average Accuracy: {avg_epoch_acc:.3f}")
        print(f"  Total Steps: {global_step}/{total_steps}")
        print(f"  Current LR: {current_lr:.2e}")

print("\n" + "="*50)
print("Training completed!")
print(f"Best training accuracy: {best_acc:.3f}")
print(f"Total steps: {global_step}")
print("="*50)

# ============================================
# TESTING PIPELINE
# ============================================
print("\nStarting testing...")

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
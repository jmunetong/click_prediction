from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
import math
import time
import json
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
import io
import os
import shutil
import gc

from model import get_model

class Trainer:
    def __init__(self, model_name, num_epochs=3,
                 batch_size=16, learning_rate=2e-5, weight_decay=0.01,
                 warmup_steps=500, max_grad_norm=1.0, log_interval=50,
                 eval_interval=500, checkpoint_dir="checkpoints", use_amp=True,
                 use_lora=False, lora_r=8, lora_alpha=16,
                 lora_dropout=0.1, lora_target_modules=None,
                 gradient_accumulation_steps=2):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        # Ensure gradient_accumulation_steps is at least 1
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.device = self._get_device()

        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["query", "value"]

        self.use_amp = use_amp
        if self.device.type not in ["cuda"]:
            self.use_amp = False
            if use_amp:
                print("Warning: Mixed precision disabled (not using CUDA)")

        self.model = self._init_model(model_name, self.device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        suffix = "_lora" if self.use_lora else ""
        # NOTE: we will save *small* model-only artifacts alongside these names
        self.best_model_path = self.checkpoint_dir / f"best_model{suffix}.pt"
        self.last_checkpoint_path = self.checkpoint_dir / f"last_model{suffix}.pt"
        self.metrics_path = self.checkpoint_dir / f"training_metrics{suffix}.json"
        self.metrics_parquet_path = self.checkpoint_dir / f"training_metrics{suffix}.parquet"

        self.optim = None
        self.scheduler = None
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.best_val_acc = 0.0
        self.epoch_train_accuracies = []
        self.epoch_train_losses = []
        self.epoch_val_accuracies = []
        self.epoch_val_losses = []

        self.step_train_losses = []
        self.step_train_accuracies = []
        self.step_numbers = []

        self.epoch_times = []

        # prefer safetensors if available
        try:
            import safetensors.torch as _st
            self._safetensors = _st
            self._use_safetensors = True
        except Exception:
            self._safetensors = None
            self._use_safetensors = False

    # ------------------------------------------------------------
    # Saving helpers (small & robust)
    # ------------------------------------------------------------
    def _cpu_state_dict(self):
        """Get model state dict on CPU to reduce GPU memory pressure & file size."""
        cpu_state = {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()}
        return cpu_state

    def _atomic_save_bytes(self, data: bytes, path: Path):
        """Write bytes atomically to avoid partial files."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _save_model_only(self, dest_path: Path, extra_metadata: dict | None = None):
        """
        Save only model weights (CPU). Prefer safetensors; fallback to torch.save(state_dict).
        """
        self.model.eval()
        cpu_sd = self._cpu_state_dict()
        try:
            if self._use_safetensors:
                meta = extra_metadata or {}
                # safetensors writes directly to file path; wrap in atomic replace
                tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
                self._safetensors.save_file(cpu_sd, str(tmp), metadata={k: str(v) for k, v in meta.items()})
                os.replace(tmp, dest_path)
            else:
                # torch.save to bytes, then atomic write
                buffer = io.BytesIO()
                torch.save(cpu_sd, buffer)
                self._atomic_save_bytes(buffer.getvalue(), dest_path)
            print(f" Saved model weights to: {dest_path}")
        except Exception as e:
            print(f"✗ Failed to save model weights to {dest_path}: {e}")
            # last-resort: minimal emergency checkpoint (very small)
            self._save_best_checkpoint_minimal(dest_path)
        finally:
            # Clean up CPU state dict from memory
            del cpu_sd
            gc.collect()

    def _save_best_checkpoint_minimal(self, dest_path: Path):
        """Emergency tiny file to indicate the best epoch/acc if full save fails."""
        try:
            tiny = {
                "note": "minimal_checkpoint_due_to_io_error",
                "best_val_acc": float(self.best_val_acc),
            }
            buf = io.BytesIO()
            torch.save(tiny, buf)
            self._atomic_save_bytes(buf.getvalue(), dest_path.with_suffix(".minimal.pt"))
            print(f"⚠ Wrote minimal fallback checkpoint: {dest_path.with_suffix('.minimal.pt')}")
        except Exception as e:
            print(f"✗ Also failed to write minimal checkpoint: {e}")

    # ------------------------------------------------------------

    def _get_device(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple GPU) device")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
        return device

    def _setup_lora(self, model):
        print("\n" + "="*50)
        print("Configuring LoRA...")
        print("="*50)
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        print("="*50 + "\n")
        return model

    def _init_model(self, model_name, device):
        model = get_model(model_name).to(device)
        if self.use_lora:
            model = self._setup_lora(model)
        # memory helper during training
        if hasattr(model, "backbone"):
            try:
                model.backbone.config.use_cache = False
                model.backbone.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing.")
            except Exception:
                pass
        return model

    def _init_optimizer(self, model, total_steps):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if self.use_lora:
            optim = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            print(f"Optimizer: Training {len(trainable_params)} parameter groups (LoRA adapters)")
        else:
            encoder_params, head_params = [], []
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    encoder_params.append(param)
                else:
                    head_params.append(param)
            optim = torch.optim.AdamW([
                {'params': encoder_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay},
                {'params': head_params, 'lr': self.learning_rate * 5, 'weight_decay': 0.0}
            ], lr=self.learning_rate, weight_decay=self.weight_decay)
            print("Optimizer: Training all parameters with differential learning rates")

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optim, lr_lambda)
        return optim, scheduler

    def test(self, test_loader):
        test_loss, test_acc = self.validate(test_loader)
        print(f"\nTest Results - Loss: {test_loss:.4f}, Accuracy:{test_acc:.4f}")
        return test_loss, test_acc


    def validate(self, val_loader):
        """Validate with proper memory management."""
        if val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention = batch["attention_mask"].to(self.device)
                cand_mask = batch["candidate_mask"].to(self.device)
                labels = batch["label_index"].to(self.device)
                
                B, N, L = input_ids.shape
                input_ids_flat = input_ids.reshape(B * N, L)
                attention_flat = attention.reshape(B * N, L)
                
                with autocast(device_type='cuda', enabled=self.use_amp):
                    logits_flat = self.model(input_ids_flat, attention_flat)
                    logits = logits_flat.view(B, N)
                    logits = logits.masked_fill(~cand_mask, float("-inf"))
                    loss = F.cross_entropy(logits, labels)
                
                pred = logits.argmax(dim=1)
                correct = (pred == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += B
                num_batches += 1
                
                # Clean up batch tensors
                del input_ids, attention, cand_mask, labels
                del input_ids_flat, attention_flat, logits_flat, logits, loss, pred
        
        # Clear cache after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        print(f"\nValidation Results:")
        print(f"  Batches: {num_batches}")
        print(f"  Samples: {total_samples}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.4f}")
        
        self.model.train()
        return avg_acc, avg_loss

    def get_metrics(self):
        """Return training metrics."""
        return {
            "epoch_train_losses": self.epoch_train_losses,
            "epoch_train_accuracies": self.epoch_train_accuracies,
            "epoch_val_losses": self.epoch_val_losses,
            "epoch_val_accuracies": self.epoch_val_accuracies,
            "step_train_losses": self.step_train_losses,
            "step_train_accuracies": self.step_train_accuracies,
            "step_numbers": self.step_numbers,
            "best_val_acc": self.best_val_acc,
            "epoch_times": self.epoch_times
        }

    def save_metrics(self):
        """Save metrics to JSON and Parquet."""
        metrics = self.get_metrics()
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to {self.metrics_path}")
        except Exception as e:
            print(f"Failed to save JSON metrics: {e}")
        
        try:
            df = pd.DataFrame({
                'epoch': list(range(1, len(self.epoch_train_losses) + 1)),
                'train_loss': self.epoch_train_losses,
                'train_acc': self.epoch_train_accuracies,
                'val_loss': self.epoch_val_losses,
                'val_acc': self.epoch_val_accuracies,
                'epoch_time': self.epoch_times
            })
            df.to_parquet(self.metrics_parquet_path)
            print(f" Saved metrics to {self.metrics_parquet_path}")
        except Exception as e:
            print(f"✗ Failed to save Parquet metrics: {e}")

    def run_train(self, train_dataset, train_loader, val_loader=None):
        total_samples = train_dataset.get_total_samples()
        print(f"Total training samples: {total_samples}")
        estimated_steps_per_epoch = total_samples // self.batch_size
        # Adjust total steps for gradient accumulation
        total_steps = self.num_epochs * (estimated_steps_per_epoch // self.gradient_accumulation_steps)
        print(f"Estimated steps per epoch: {estimated_steps_per_epoch}")
        print(f"Total optimizer steps: {total_steps}")
        print(f"Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print("Training Mode:", "LoRA" if self.use_lora else "Full Fine-tuning")

        self.optim, self.scheduler = self._init_optimizer(self.model, total_steps)

        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Warmup Steps: {self.warmup_steps}")
        print(f"Gradient Clipping: {self.max_grad_norm}")
        print("="*50)

        global_step = 0
        total_training_time = 0.0

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            epoch_start_time = time.time()

            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            batch_count = 0
            accumulated_loss = 0.0
            
            # Reset gradients at epoch start
            self.optim.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader):
                if epoch_total >= total_samples:
                    print(f"\nCompleted epoch {epoch + 1} (processed {epoch_total} samples)")
                    break

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention = batch["attention_mask"].to(self.device, non_blocking=True)
                cand_mask = batch["candidate_mask"].to(self.device, non_blocking=True)
                labels = batch["label_index"].to(self.device, non_blocking=True)

                B, N, L = input_ids.shape
                input_ids_flat = input_ids.reshape(B * N, L)
                attention_flat = attention.reshape(B * N, L)

                # Forward pass with mixed precision
                with autocast(device_type='cuda', enabled=self.use_amp):
                    logits_flat = self.model(input_ids_flat, attention_flat)
                    logits = logits_flat.view(B, N)
                    logits = logits.masked_fill(~cand_mask, float("-inf"))
                    loss = F.cross_entropy(logits, labels)
                    
                    # Scale loss by accumulation steps for gradient accumulation
                    scaled_loss = loss / self.gradient_accumulation_steps

                # Backward pass with gradient scaling
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Accumulate metrics (using original unscaled loss)
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    correct = (pred == labels).sum().item()
                    batch_acc = correct / B
                    epoch_correct += correct
                    epoch_total += B
                    batch_loss = loss.item()
                    accumulated_loss += batch_loss
                    epoch_loss += batch_loss
                    batch_count += 1

                # Update weights every accumulation_steps batches OR at end of epoch
                is_accumulation_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
                is_last_batch = epoch_total >= total_samples
                
                if is_accumulation_step or is_last_batch:
                    if self.use_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optim.step()
                    
                    self.optim.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    
                    # Get current learning rate
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # Log averaged metrics over accumulation window
                    num_accumulated = min(self.gradient_accumulation_steps, batch_count % self.gradient_accumulation_steps or self.gradient_accumulation_steps)
                    avg_accumulated_loss = accumulated_loss / num_accumulated
                    
                    self.step_train_losses.append(avg_accumulated_loss)
                    self.step_train_accuracies.append(batch_acc)
                    self.step_numbers.append(global_step)
                    
                    if global_step % self.log_interval == 0:
                        print(f"epoch {epoch + 1} | step {global_step} | batch {batch_idx} | "
                              f"samples {epoch_total}/{total_samples} | "
                              f"loss {avg_accumulated_loss:.4f} | "
                              f"acc {batch_acc:.3f} | lr {current_lr:.2e}")
                    
                    global_step += 1
                    accumulated_loss = 0.0  # Reset for next accumulation window

                # Clean up batch tensors to free memory
                del input_ids, attention, cand_mask, labels
                del input_ids_flat, attention_flat, logits_flat, logits, loss, scaled_loss, pred
                
                # Periodic garbage collection
                if batch_idx % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            # Apply any remaining accumulated gradients at end of epoch
            # This handles cases where total batches don't divide evenly by accumulation steps
            if accumulated_loss > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optim.step()
                
                self.optim.zero_grad(set_to_none=True)
                self.scheduler.step()
                print(f"Applied final accumulated gradients for epoch {epoch + 1}")

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            total_training_time += epoch_time

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                avg_epoch_acc = epoch_correct / epoch_total
                self.epoch_train_accuracies.append(avg_epoch_acc)
                self.epoch_train_losses.append(avg_epoch_loss)

                print(f"\nEpoch {epoch + 1} Training Summary:")
                print(f"  Batches processed: {batch_count}")
                print(f"  Samples processed: {epoch_total}")
                print(f"  Average Loss: {avg_epoch_loss:.4f}")
                print(f"  Average Accuracy: {avg_epoch_acc:.3f}")
                print(f"  Epoch Time: {epoch_time:.2f}s")

                # Validate every epoch with proper memory cleanup
                print("\nRunning validation...")
                val_acc, val_loss = self.validate(val_loader)
                self.epoch_val_accuracies.append(val_acc)
                self.epoch_val_losses.append(val_loss)

                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    print(f" New best validation accuracy: {val_acc:.4f}")
                    meta = {"epoch": epoch + 1, "best_val_acc": val_acc}
                    self._save_model_only(self.best_model_path, extra_metadata=meta)
                    if self.use_lora:
                        # save adapters separately if desired
                        adapter_path = self.checkpoint_dir / f"best_lora_adapters"
                        try:
                            self.model.save_pretrained(adapter_path)
                            print(f" LoRA adapters saved to {adapter_path}")
                        except Exception as e:
                            print(f"⚠ Failed to save LoRA adapters: {e}")
                
                # Clear cache after epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        avg_epoch_time = total_training_time / len(self.epoch_times) if self.epoch_times else 0

        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print("\nTiming Statistics:")
        for i, t in enumerate(self.epoch_times, 1):
            print(f"  Epoch {i}: {t:.2f}s")
        
        self.save_metrics()
        return self.get_metrics()
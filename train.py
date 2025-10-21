from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import math
import time
import json
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType

from model import CrossEncoderScorer

class Trainer:

    def __init__(self, num_epochs=3, 
                 batch_size=16, learning_rate=2e-5, weight_decay=0.01,
                 warmup_steps=500, max_grad_norm=1.0, log_interval=50, 
                 eval_interval=500, checkpoint_dir="checkpoints", use_amp=True,
                 use_lora=False, lora_r=8, lora_alpha=16, 
                 lora_dropout=0.1, lora_target_modules=None):
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.device = self._get_device()
        
        # LoRA configuration (QLoRA removed)
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["query", "value"]  # Default for RoBERTa
        
        # Mixed precision setup
        self.use_amp = use_amp
        if self.device.type not in ["cuda"]:
            self.use_amp = False
            if use_amp:
                print("Warning: Mixed precision disabled (not using CUDA)")
        
        self.model = self._init_model(self.device)
        
        # Update checkpoint paths based on LoRA usage
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        suffix = "_lora" if self.use_lora else ""
        
        self.best_model_path = self.checkpoint_dir / f"best_model{suffix}.pt"
        self.last_checkpoint_path = self.checkpoint_dir / f"last_model{suffix}.pt"
        self.metrics_path = self.checkpoint_dir / f"training_metrics{suffix}.json"
        self.metrics_parquet_path = self.checkpoint_dir / f"training_metrics{suffix}.parquet"
        
        # Will be initialized in run_train when we know total_steps
        self.optim = None
        self.scheduler = None
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Tracking metrics
        self.best_val_acc = 0.0
        self.epoch_train_accuracies = []
        self.epoch_train_losses = []
        self.epoch_val_accuracies = []
        self.epoch_val_losses = []
        
        # Step-level metrics (for detailed plots)
        self.step_train_losses = []
        self.step_train_accuracies = []
        self.step_numbers = []
        
        # Time tracking
        self.epoch_times = []

    def _get_device(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple GPU) device")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
        return device

    def _setup_lora(self, model):
        """Configure and apply LoRA to the model"""
        print("\n" + "="*50)
        print("Configuring LoRA...")
        print("="*50)
        
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION  # For encoding tasks
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"LoRA Configuration:")
        print(f"  r: {self.lora_r}")
        print(f"  alpha: {self.lora_alpha}")
        print(f"  dropout: {self.lora_dropout}")
        print(f"  target_modules: {self.lora_target_modules}")
        print(f"Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        print("="*50 + "\n")
        
        return model

    def _init_model(self, device):
        if self.use_lora:
            # Standard LoRA without quantization
            model = CrossEncoderScorer("roberta-base").to(device)
            model = self._setup_lora(model)
        else:
            # Standard full fine-tuning
            model = CrossEncoderScorer("roberta-base").to(device)
        return model
    
    def _init_optimizer(self, model, total_steps):
        # Get trainable parameters only (important for LoRA)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if self.use_lora:
            # For LoRA, only optimize the LoRA parameters
            optim = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            print(f"Optimizer: Training {len(trainable_params)} parameter groups (LoRA adapters)")
        else:
            # Standard full fine-tuning with differential learning rates
            encoder_params = []
            head_params = []
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    encoder_params.append(param)
                else:
                    head_params.append(param)

            optim = torch.optim.AdamW([
                {'params': encoder_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay},
                {'params': head_params, 'lr': self.learning_rate * 5, 'weight_decay': 0.0}
            ], lr=self.learning_rate, weight_decay=self.weight_decay)
            print(f"Optimizer: Training all parameters with differential learning rates")

        # Learning rate scheduler with warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                progress = float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optim, lr_lambda)
        return optim, scheduler
    
    def validate(self, val_loader):
        """Run validation and return metrics"""
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        
        print("Running validation...")
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention = batch["attention_mask"].to(self.device)
                cand_mask = batch["candidate_mask"].to(self.device)
                labels = batch["label_index"].to(self.device)

                B, N, L = input_ids.shape
                input_ids_flat = input_ids.reshape(B * N, L)
                attention_flat = attention.reshape(B * N, L)

                # Use autocast for validation too
                with autocast(device_type='cuda', enabled=self.use_amp):
                    logits_flat = self.model(input_ids_flat, attention_flat)
                    logits = logits_flat.view(B, N)
                    logits = logits.masked_fill(~cand_mask, float("-inf"))
                    loss = F.cross_entropy(logits, labels)
                
                pred = logits.argmax(dim=1)
                
                total_correct += (pred == labels).sum().item()
                total_samples += B
                total_loss += loss.item()
                num_batches += 1
        
        val_acc = total_correct / total_samples if total_samples > 0 else 0.0
        val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.4f} ({total_correct}/{total_samples})")
        
        return val_acc, val_loss
    
    def test(self, test_loader):
        """Run testing on test dataset - wrapper around validate()"""
        print("\n" + "="*50)
        print("Starting testing on test set...")
        print("="*50)
        
        test_acc, test_loss = self.validate(test_loader)
        
        print("\n" + "="*50)
        print("TESTING RESULTS")
        print("="*50)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("="*50)
        
        return test_acc, test_loss
    
    def get_metrics(self):
        """Return all stored metrics for plotting"""
        return {
            'epoch_train_accuracies': self.epoch_train_accuracies,
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_val_accuracies': self.epoch_val_accuracies,
            'epoch_val_losses': self.epoch_val_losses,
            'step_train_losses': self.step_train_losses,
            'step_train_accuracies': self.step_train_accuracies,
            'step_numbers': self.step_numbers,
            'best_val_acc': self.best_val_acc,
            'epoch_times': self.epoch_times,
        }
    
    def save_metrics(self):
        """Save all training metrics to JSON and Parquet files"""
        print("\n" + "="*50)
        print("Saving training metrics...")
        print("="*50)
        
        # Determine training mode (QLoRA removed)
        training_mode = "LoRA" if self.use_lora else "Full Fine-tuning"
        
        # Create metrics dictionary
        metrics_dict = {
            'training_mode': training_mode,
            'num_epochs': len(self.epoch_train_accuracies),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'use_amp': self.use_amp,
            'use_lora': self.use_lora,
            'epoch_train_accuracies': self.epoch_train_accuracies,
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_val_accuracies': self.epoch_val_accuracies,
            'epoch_val_losses': self.epoch_val_losses,
            'best_val_acc': self.best_val_acc,
            'epoch_times': self.epoch_times,
            'total_training_time': sum(self.epoch_times) if self.epoch_times else 0,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
        }
        
        # Save to JSON
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✓ Saved metrics to JSON: {self.metrics_path}")
        
        # Save epoch-level metrics to Parquet
        epochs = list(range(1, len(self.epoch_train_accuracies) + 1))
        
        df_data = {
            'epoch': epochs,
            'train_accuracy': self.epoch_train_accuracies,
            'train_loss': self.epoch_train_losses,
            'training_mode': [training_mode] * len(epochs),
            'dataset_train': ['click_train.parquet'] * len(epochs),
        }
        
        # Add validation metrics if available
        if self.epoch_val_accuracies:
            df_data['val_accuracy'] = self.epoch_val_accuracies
            df_data['val_loss'] = self.epoch_val_losses
        
        # Add timing metrics if available
        if self.epoch_times:
            df_data['epoch_time_seconds'] = self.epoch_times
        
        df_metrics = pd.DataFrame(df_data)
        df_metrics.to_parquet(self.metrics_parquet_path, index=False, engine='pyarrow')
        print(f"✓ Saved epoch metrics to Parquet: {self.metrics_parquet_path}")
        
        # Save step-level metrics to Parquet if available
        if self.step_train_losses and len(self.step_train_losses) > 0:
            step_parquet_path = self.checkpoint_dir / f"step_metrics{'.lora' if self.use_lora else ''}.parquet"
            df_steps = pd.DataFrame({
                'step': self.step_numbers,
                'train_loss': self.step_train_losses,
                'train_accuracy': self.step_train_accuracies,
                'training_mode': [training_mode] * len(self.step_numbers),
                'dataset': ['click_train.parquet'] * len(self.step_numbers),
            })
            df_steps.to_parquet(step_parquet_path, index=False, engine='pyarrow')
            print(f"✓ Saved step metrics to Parquet: {step_parquet_path}")
        
        print("="*50)
    
    def save_lora_adapters(self, path):
        """Save only LoRA adapters (for LoRA models)"""
        if not self.use_lora:
            print("Model is not using LoRA. Use standard checkpoint saving.")
            return
        
        self.model.save_pretrained(path)
        print(f"LoRA adapters saved to {path}")
    
    def _format_time(self, seconds):
        """Format seconds into human-readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def run_train(self, train_dataset, train_loader, val_loader=None):
        # Get total samples for accurate scheduler calculation
        total_samples = train_dataset.get_total_samples()
        print(f"Total training samples: {total_samples}")
        estimated_steps_per_epoch = total_samples // self.batch_size
        total_steps = self.num_epochs * estimated_steps_per_epoch
        print(f"Estimated steps per epoch: {estimated_steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        
        if self.use_lora:
            print(f"Training Mode: LoRA")
        else:
            print(f"Training Mode: Full Fine-tuning")
        
        # Initialize optimizer and scheduler now that we know total_steps
        self.optim, self.scheduler = self._init_optimizer(self.model, total_steps)
        
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Warmup Steps: {self.warmup_steps}")
        print(f"Gradient Clipping: {self.max_grad_norm}")
        print("="*50)

        global_step = 0
        training_complete = False
        total_training_time = 0.0

        for epoch in range(self.num_epochs):
            if training_complete:
                break

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)

            # Start epoch timer
            epoch_start_time = time.time()

            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Check if we've processed all training samples for this epoch
                if epoch_total >= total_samples:
                    print(f"\nCompleted epoch {epoch + 1} (processed {epoch_total} samples)")
                    break
                    
                input_ids = batch["input_ids"].to(self.device)
                attention = batch["attention_mask"].to(self.device)
                cand_mask = batch["candidate_mask"].to(self.device)
                labels = batch["label_index"].to(self.device)

                B, N, L = input_ids.shape
                input_ids_flat = input_ids.reshape(B * N, L)
                attention_flat = attention.reshape(B * N, L)

                # Forward pass with autocast
                with autocast(device_type='cuda', enabled=self.use_amp):
                    logits_flat = self.model(input_ids_flat, attention_flat)
                    logits = logits_flat.view(B, N)
                    logits = logits.masked_fill(~cand_mask, float("-inf"))
                    loss = F.cross_entropy(logits, labels)

                # Backward pass with gradient scaling
                self.optim.zero_grad(set_to_none=True)
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optim.step()
                
                self.scheduler.step()

                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]

                # Track metrics
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    correct = (pred == labels).sum().item()
                    batch_acc = correct / B
                    
                    epoch_correct += correct
                    epoch_total += B
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Store step-level metrics
                    self.step_train_losses.append(loss.item())
                    self.step_train_accuracies.append(batch_acc)
                    self.step_numbers.append(global_step)

                if global_step % self.log_interval == 0:
                    print(f"epoch {epoch + 1} | step {global_step} | batch {batch_idx} | "
                          f"samples {epoch_total}/{total_samples} | "
                          f"loss {loss.item():.4f} | acc {batch_acc:.3f} | lr {current_lr:.2e}")
                
                global_step += 1
            
            # End epoch timer
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            total_training_time += epoch_time
            
            # End of epoch: compute metrics and validate
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                avg_epoch_acc = epoch_correct / epoch_total
                
                # STORE EPOCH-LEVEL METRICS (ALWAYS - FOR ALL EPOCHS)
                self.epoch_train_accuracies.append(avg_epoch_acc)
                self.epoch_train_losses.append(avg_epoch_loss)
                
                print(f"\nEpoch {epoch + 1} Training Summary:")
                print(f"  Batches processed: {batch_count}")
                print(f"  Samples processed: {epoch_total}")
                print(f"  Average Loss: {avg_epoch_loss:.4f}")
                print(f"  Average Accuracy: {avg_epoch_acc:.3f}")
                print(f"  Current LR: {current_lr:.2e}")
                print(f"  Epoch Time: {self._format_time(epoch_time)} ({epoch_time:.2f}s)")
                
                # Run validation if provided
         
                val_acc, val_loss = self.validate(val_loader)
                
                # STORE VALIDATION METRICS (ALWAYS - FOR ALL EPOCHS)
                self.epoch_val_accuracies.append(val_acc)
                self.epoch_val_losses.append(val_loss)
                
                # Save best model based on validation accuracy
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    
                    # Save checkpoint with FULL history
                    checkpoint = {
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_accuracy': avg_epoch_acc,
                        'train_loss': avg_epoch_loss,
                        'val_accuracy': val_acc,
                        'val_loss': val_loss,
                        'epoch_train_accuracies': self.epoch_train_accuracies,  # ALL epochs
                        'epoch_train_losses': self.epoch_train_losses,          # ALL epochs
                        'epoch_val_accuracies': self.epoch_val_accuracies,      # ALL epochs
                        'epoch_val_losses': self.epoch_val_losses,              # ALL epochs
                        'step_train_losses': self.step_train_losses,
                        'step_train_accuracies': self.step_train_accuracies,
                        'step_numbers': self.step_numbers,
                        'epoch_times': self.epoch_times,
                        'use_lora': self.use_lora,
                    }
                    
                    if self.use_amp:
                        checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                    
                    torch.save(checkpoint, self.best_model_path)
                    print(f"  → New best model saved! val_acc={val_acc:.4f}")
                    
                    # Also save LoRA adapters separately if using LoRA
                    if self.use_lora:
                        adapter_path = self.checkpoint_dir / f"best_lora_adapters"
                        self.save_lora_adapters(adapter_path)
                

        # Calculate timing statistics
        avg_epoch_time = total_training_time / len(self.epoch_times) if self.epoch_times else 0
        
        print("\n" + "="*50)
        print("Training completed!")
        if val_loader is not None:
            print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        else:
            print(f"Best training accuracy: {self.best_val_acc:.4f}")
        print(f"Total steps: {global_step}")
        print("\nTiming Statistics:")
        print(f"  Total training time: {self._format_time(total_training_time)} ({total_training_time:.2f}s)")
        print(f"  Average time per epoch: {self._format_time(avg_epoch_time)} ({avg_epoch_time:.2f}s)")
        if self.epoch_times:
            print(f"  Fastest epoch: {self._format_time(min(self.epoch_times))} ({min(self.epoch_times):.2f}s)")
            print(f"  Slowest epoch: {self._format_time(max(self.epoch_times))} ({max(self.epoch_times):.2f}s)")
        print("="*50)
        
        # Save all metrics to separate files
        self.save_metrics()
        
        # Return comprehensive metrics
        return self.get_metrics()
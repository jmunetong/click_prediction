"""
plot_results.py

Standalone script for plotting training, validation, and test results.
Emphasis on training (from preprocessed_click_train.parquet) vs test (from preprocessed_click_test.parquet)

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from pathlib import Path
import json


def plot_training_results(
    epoch_train_accuracies,
    epoch_train_losses,
    epoch_val_accuracies=None,
    epoch_val_losses=None,
    test_acc=None,
    test_loss=None,
    save_dir="plots",
    suffix="",
    step_train_losses=None,
    step_train_accuracies=None,
    step_numbers=None,
    best_val_acc=None,
    training_mode="Full Fine-tuning",
    num_epochs=None,
    batch_size=None,
    learning_rate=None
):
    """
    Plot training and validation metrics, with test results
    
    Note: Training data comes from click_train.parquet (preprocessed)
          Test data comes from click_test.parquet (preprocessed)
    
    Args:
        epoch_train_accuracies: List of training accuracies per epoch
        epoch_train_losses: List of training losses per epoch
        epoch_val_accuracies: List of validation accuracies per epoch (optional)
        epoch_val_losses: List of validation losses per epoch (optional)
        test_acc: Test accuracy from click_test.parquet (REQUIRED for assignment)
        test_loss: Test loss from click_test.parquet (REQUIRED for assignment)
        save_dir: Directory to save plots
        suffix: Suffix for filenames (e.g., "_lora")
        step_train_losses: List of training losses per step (optional)
        step_train_accuracies: List of training accuracies per step (optional)
        step_numbers: List of step numbers (optional)
        best_val_acc: Best validation accuracy achieved (optional)
        training_mode: String describing training mode (optional)
        num_epochs: Number of epochs (optional)
        batch_size: Batch size used (optional)
        learning_rate: Learning rate used (optional)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Ensure we use ALL epochs from the data
    num_epochs_actual = len(epoch_train_accuracies)
    epochs = np.arange(1, num_epochs_actual + 1)
    has_validation = epoch_val_accuracies is not None and len(epoch_val_accuracies) > 0
    
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    print(f"Save directory: {save_dir}")
    print(f"Training data source: click_train.parquet (preprocessed)")
    print(f"Test data source: click_test.parquet (preprocessed)")
    print(f"Number of epochs to plot: {num_epochs_actual}")
    if test_acc is not None:
        print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        if test_acc >= 0.40:
            print("✓ Test accuracy meets requirement (≥40%)")
        else:
            print("⚠ Warning: Test accuracy below expected 40%")
    print("="*60)
    
    # ========================================
    # Plot 1: PRIMARY PLOT - Training vs Test Accuracy
    # This is the key plot for the assignment requirement
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot training accuracy per epoch (ALL EPOCHS)
    ax.plot(epochs, epoch_train_accuracies, 'o-', linewidth=3, markersize=10,
            label='Training Accuracy (click_train.parquet)', color='#2E86AB')
    
    # Add validation if available (ALL EPOCHS)
    if has_validation:
        ax.plot(epochs, epoch_val_accuracies, 's-', linewidth=2.5, markersize=9,
                label='Validation Accuracy', color='#A23B72', alpha=0.8)
    
    # Add test accuracy prominently
    if test_acc is not None:
        # Horizontal line
        ax.axhline(y=test_acc, color='#F18F01', linestyle='--', linewidth=3,
                   label=f'Test Accuracy (click_test.parquet): {test_acc:.4f}')
        # Diamond marker at the end for emphasis
        ax.plot(epochs[-1], test_acc, 'D', markersize=15, color='#F18F01', 
                markeredgewidth=2, markeredgecolor='darkred')
        
        # Add text annotation for test accuracy
        ax.text(epochs[-1] * 0.98, test_acc, f'  {test_acc:.4f} ({test_acc*100:.1f}%)',
                fontsize=12, fontweight='bold', color='#F18F01',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#F18F01', linewidth=2))
    
    # Add 40% threshold line
    ax.axhline(y=0.40, color='gray', linestyle=':', linewidth=2, alpha=0.5,
               label='Expected threshold (40%)')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Training and Test Accuracy (All {num_epochs_actual} Epochs)\n(click_train.parquet vs click_test.parquet)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    ax.set_xticks(epochs)
    
    # Add percentage labels on y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_test_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✓ SAVED PRIMARY PLOT: {save_dir / 'training_test_accuracy.png'}")
    plt.close()
    
    # ========================================
    # Plot 2: Training and Test Loss
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot epoch-level losses (ALL EPOCHS)
    ax.plot(epochs, epoch_train_losses, 'o-', linewidth=3, markersize=10,
            label='Training Loss (click_train.parquet)', color='#2E86AB')
    
    if has_validation:
        ax.plot(epochs, epoch_val_losses, 's-', linewidth=2.5, markersize=9,
                label='Validation Loss', color='#A23B72', alpha=0.8)
    
    # Add test loss
    if test_loss is not None:
        ax.axhline(y=test_loss, color='#F18F01', linestyle='--', linewidth=3,
                   label=f'Test Loss (click_test.parquet): {test_loss:.4f}')
        ax.plot(epochs[-1], test_loss, 'D', markersize=15, color='#F18F01',
                markeredgewidth=2, markeredgecolor='darkred')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'Training and Test Loss (All {num_epochs_actual} Epochs)\n(click_train.parquet vs click_test.parquet)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_test_loss.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'training_test_loss.png'}")
    plt.close()
    
    # ========================================
    # Plot 3: Combined Plot (Loss and Accuracy side by side)
    # ========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Loss subplot (ALL EPOCHS)
    ax1.plot(epochs, epoch_train_losses, 'o-', linewidth=2.5, markersize=9,
             label='Training Loss', color='#2E86AB')
    if has_validation:
        ax1.plot(epochs, epoch_val_losses, 's-', linewidth=2, markersize=8,
                 label='Validation Loss', color='#A23B72', alpha=0.8)
    if test_loss is not None:
        ax1.axhline(y=test_loss, color='#F18F01', linestyle='--', linewidth=2.5,
                    label=f'Test Loss ({test_loss:.4f})')
        ax1.plot(epochs[-1], test_loss, 'D', markersize=12, color='#F18F01')
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Loss per Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(epochs)
    
    # Accuracy subplot (ALL EPOCHS)
    ax2.plot(epochs, epoch_train_accuracies, 'o-', linewidth=2.5, markersize=9,
             label='Training Accuracy', color='#2E86AB')
    if has_validation:
        ax2.plot(epochs, epoch_val_accuracies, 's-', linewidth=2, markersize=8,
                 label='Validation Accuracy', color='#A23B72', alpha=0.8)
    if test_acc is not None:
        ax2.axhline(y=test_acc, color='#F18F01', linestyle='--', linewidth=2.5,
                    label=f'Test Accuracy ({test_acc:.4f})')
        ax2.plot(epochs[-1], test_acc, 'D', markersize=12, color='#F18F01')
    
    # Add 40% threshold
    ax2.axhline(y=0.40, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                label='40% threshold')
    
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Accuracy per Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.0])
    ax2.set_xticks(epochs)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    fig.suptitle(f'Training vs Test Performance - All {num_epochs_actual} Epochs\n(click_train.parquet vs click_test.parquet)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'combined_plot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'combined_plot.png'}")
    plt.close()
    
    # ========================================
    # Plot 4: Step-level Training Progress (optional, for detailed analysis)
    # ========================================
    if step_train_losses is not None and len(step_train_losses) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss over steps
        ax1.plot(step_numbers, step_train_losses, alpha=0.3, color='#2E86AB', linewidth=0.5)
        
        # Add smoothed line
        window_size = min(50, len(step_train_losses) // 10)
        if window_size > 1:
            smoothed_loss = np.convolve(step_train_losses, 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
            ax1.plot(step_numbers[window_size-1:], smoothed_loss, 
                    color='#2E86AB', linewidth=2, label=f'Smoothed (window={window_size})')
        
        ax1.set_xlabel('Training Step', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax1.set_title('Training Loss (Step-level Detail)', fontsize=14, fontweight='bold')
        if window_size > 1:
            ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Accuracy over steps
        ax2.plot(step_numbers, step_train_accuracies, alpha=0.3, color='#2E86AB', linewidth=0.5)
        
        # Add smoothed line
        if window_size > 1:
            smoothed_acc = np.convolve(step_train_accuracies, 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
            ax2.plot(step_numbers[window_size-1:], smoothed_acc, 
                    color='#2E86AB', linewidth=2, label=f'Smoothed (window={window_size})')
        
        ax2.set_xlabel('Training Step', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax2.set_title('Training Accuracy (Step-level Detail)', fontsize=14, fontweight='bold')
        if window_size > 1:
            ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_dir / 'step_level_plot.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir / 'step_level_plot.png'}")
        plt.close()
    
    print("\n" + "="*60)
    print("All plots saved successfully!")
    print("="*60)
    
    # Print final summary
    if test_acc is not None:
        print(f"\n📊 FINAL TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
        if test_acc >= 0.40:
            print("✓ Test accuracy meets the expected threshold of 40%")
        else:
            print(f"⚠ Test accuracy is below expected 40% (gap: {(0.40-test_acc)*100:.2f}%)")
    
    print(f"\n✓ Plotted all {num_epochs_actual} epochs")


def load_checkpoint_and_plot(checkpoint_path, test_acc=None, test_loss=None, save_dir="plots"):
    """
    Load checkpoint and create plots
    
    Args:
        checkpoint_path: Path to checkpoint file
        test_acc: Test accuracy from click_test.parquet (REQUIRED)
        test_loss: Test loss from click_test.parquet (REQUIRED)
        save_dir: Base directory to save plots (will be appended with suffix)
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract metrics from checkpoint (ALWAYS USE ALL AVAILABLE EPOCHS)
    epoch_train_accuracies = checkpoint.get('epoch_train_accuracies', [])
    epoch_train_losses = checkpoint.get('epoch_train_losses', [])
    epoch_val_accuracies = checkpoint.get('epoch_val_accuracies', [])
    epoch_val_losses = checkpoint.get('epoch_val_losses', [])
    step_train_losses = checkpoint.get('step_train_losses', None)
    step_train_accuracies = checkpoint.get('step_train_accuracies', None)
    step_numbers = checkpoint.get('step_numbers', None)
    best_val_acc = checkpoint.get('val_accuracy', checkpoint.get('train_accuracy', None))
    
    # Determine training mode and suffix
    use_lora = checkpoint.get('use_lora', False)
    
    if use_lora:
        suffix = "_lora"
        training_mode = "LoRA"
    else:
        suffix = ""
        training_mode = "Full Fine-tuning"
    
    # Create save directory with suffix
    base_save_dir = Path(save_dir)
    if suffix:
        # If suffix exists (e.g., "_lora"), create subdirectory
        actual_save_dir = base_save_dir / training_mode.lower().replace(" ", "_")
    else:
        # For full fine-tuning, use "full_finetuning" subdirectory
        actual_save_dir = base_save_dir / "full_finetuning"
    
    # Get other metadata
    num_epochs = len(epoch_train_accuracies)  # Use actual number from data
    
    print(f"Training mode: {training_mode}")
    print(f"Total epochs in checkpoint: {num_epochs}")
    print(f"Train accuracies available: {len(epoch_train_accuracies)}")
    print(f"Val accuracies available: {len(epoch_val_accuracies)}")
    
    if test_acc is None or test_loss is None:
        print("\n⚠ WARNING: Test accuracy and/or test loss not provided!")
        print("   For assignment requirements, you must provide test results from click_test.parquet")
    
    # Create plots with ALL epochs
    plot_training_results(
        epoch_train_accuracies=epoch_train_accuracies,
        epoch_train_losses=epoch_train_losses,
        epoch_val_accuracies=epoch_val_accuracies if len(epoch_val_accuracies) > 0 else None,
        epoch_val_losses=epoch_val_losses if len(epoch_val_losses) > 0 else None,
        test_acc=test_acc,
        test_loss=test_loss,
        save_dir=actual_save_dir,
        suffix="",  # No suffix in filenames since directory already has it
        step_train_losses=step_train_losses,
        step_train_accuracies=step_train_accuracies,
        step_numbers=step_numbers,
        best_val_acc=best_val_acc,
        training_mode=training_mode,
        num_epochs=num_epochs,
    )
    
    # Save metrics to JSON in the same directory
    metrics = {
        'training_dataset': 'click_train.parquet (preprocessed)',
        'test_dataset': 'click_test.parquet (preprocessed)',
        'num_epochs': num_epochs,
        'epoch_train_accuracies': epoch_train_accuracies,
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_accuracies': epoch_val_accuracies,
        'epoch_val_losses': epoch_val_losses,
        'best_val_acc': best_val_acc,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'meets_40_percent_requirement': test_acc >= 0.40 if test_acc is not None else None,
        'training_mode': training_mode,
    }
    
    metrics_file = actual_save_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training results from checkpoint\n'
                    'Training data: click_train.parquet\n'
                    'Test data: click_test.parquet\n'
                    'Expected test accuracy: ≥40%',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/last_model.pt)')
    parser.add_argument('--test_acc', type=float, required=True,
                       help='Test accuracy from click_test.parquet (REQUIRED)')
    parser.add_argument('--test_loss', type=float, required=True,
                       help='Test loss from click_test.parquet (REQUIRED)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Base directory to save plots (default: results)')
    
    args = parser.parse_args()
    
    load_checkpoint_and_plot(
        checkpoint_path=args.checkpoint_path,
        test_acc=args.test_acc,
        test_loss=args.test_loss,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()



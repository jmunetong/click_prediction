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
        suffix: Suffix for filenames (e.g., "_lora", "_qlora")
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
    save_dir.mkdir(exist_ok=True)
    
    epochs = np.arange(1, len(epoch_train_accuracies) + 1)
    has_validation = epoch_val_accuracies is not None and len(epoch_val_accuracies) > 0
    
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    print(f"Training data source: click_train.parquet (preprocessed)")
    print(f"Test data source: click_test.parquet (preprocessed)")
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
    
    # Plot training accuracy per epoch
    ax.plot(epochs, epoch_train_accuracies, 'o-', linewidth=3, markersize=10,
            label='Training Accuracy (click_train.parquet)', color='#2E86AB')
    
    # Add validation if available
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
    ax.set_title('Training and Test Accuracy\n(click_train.parquet vs click_test.parquet)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    ax.set_xticks(epochs)
    
    # Add percentage labels on y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_test_accuracy{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ SAVED PRIMARY PLOT: {save_dir / f'training_test_accuracy{suffix}.png'}")
    plt.close()
    
    # ========================================
    # Training and Test Loss
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot epoch-level losses
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
    ax.set_title('Training and Test Loss\n(click_train.parquet vs click_test.parquet)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_test_loss{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / f'training_test_loss{suffix}.png'}")
    plt.close()
    
    # ========================================
    # Combined Plot (Loss and Accuracy side by side)
    # ========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Loss subplot
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
    
    # Accuracy subplot
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
    
    fig.suptitle('Training vs Test Performance (click_train.parquet vs click_test.parquet)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'combined_plot{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / f'combined_plot{suffix}.png'}")
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
        plt.savefig(save_dir / f'step_level_plot{suffix}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir / f'step_level_plot{suffix}.png'}")
        plt.close()
    
    # ========================================
    # Plot 5: Summary Table with Dataset Information
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [
        ['Metric', 'Value'],
        ['', ''],  # Separator
        ['DATASET INFORMATION', ''],
        ['Training Dataset', 'click_train.parquet (preprocessed)'],
        ['Test Dataset', 'click_test.parquet (preprocessed)'],
        ['', ''],  # Separator
        ['TRAINING RESULTS', ''],
    ]
    
    if num_epochs is not None:
        table_data.append(['Total Epochs', f'{num_epochs}'])
    else:
        table_data.append(['Total Epochs', f'{len(epoch_train_accuracies)}'])
    
    table_data.extend([
        ['Final Train Loss', f'{epoch_train_losses[-1]:.4f}'],
        ['Final Train Accuracy', f'{epoch_train_accuracies[-1]:.4f} ({epoch_train_accuracies[-1]*100:.2f}%)'],
    ])
    
    if has_validation:
        table_data.extend([
            ['Final Val Loss', f'{epoch_val_losses[-1]:.4f}'],
            ['Final Val Accuracy', f'{epoch_val_accuracies[-1]:.4f} ({epoch_val_accuracies[-1]*100:.2f}%)'],
        ])
        if best_val_acc is not None:
            table_data.append(['Best Val Accuracy', f'{best_val_acc:.4f} ({best_val_acc*100:.2f}%)'])
    
    table_data.append(['', ''])  # Separator
    table_data.append(['TEST RESULTS (click_test.parquet)', ''])
    
    if test_loss is not None:
        table_data.append(['Test Loss', f'{test_loss:.4f}'])
    if test_acc is not None:
        meets_req = '✓ MEETS REQUIREMENT' if test_acc >= 0.40 else '⚠ Below expected'
        table_data.append(['Test Accuracy', f'{test_acc:.4f} ({test_acc*100:.2f}%) {meets_req}'])
        table_data.append(['Expected Threshold', '≥40%'])
    
    table_data.append(['', ''])  # Separator
    table_data.append(['CONFIGURATION', ''])
    if training_mode:
        table_data.append(['Training Mode', training_mode])
    if learning_rate is not None:
        table_data.append(['Learning Rate', f'{learning_rate}'])
    if batch_size is not None:
        table_data.append(['Batch Size', f'{batch_size}'])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style section headers (rows with empty second column or section titles)
    section_rows = [2, 6, 10, 15]  # Adjust based on actual row indices
    for row_idx in section_rows:
        if row_idx < len(table_data):
            for j in range(2):
                table[(row_idx, j)].set_facecolor('#A8DADC')
                table[(row_idx, j)].set_text_props(weight='bold')
    
    # Alternate row colors for data rows
    for i in range(1, len(table_data)):
        if i not in section_rows and table_data[i][1] != '':
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F1FAEE')
    
    plt.title('Training Summary: click_train.parquet vs click_test.parquet', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_dir / f'summary_table{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / f'summary_table{suffix}.png'}")
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


def load_checkpoint_and_plot(checkpoint_path, test_acc=None, test_loss=None, save_dir="plots"):
    """
    Load checkpoint and create plots
    
    Args:
        checkpoint_path: Path to checkpoint file
        test_acc: Test accuracy from click_test.parquet (REQUIRED)
        test_loss: Test loss from click_test.parquet (REQUIRED)
        save_dir: Directory to save plots
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract metrics from checkpoint
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
    use_qlora = checkpoint.get('use_qlora', False)
    
    if use_qlora:
        suffix = "_qlora"
        training_mode = "QLoRA"
    elif use_lora:
        suffix = "_lora"
        training_mode = "LoRA"
    else:
        suffix = ""
        training_mode = "Full Fine-tuning"
    
    # Get other metadata
    num_epochs = checkpoint.get('epoch', len(epoch_train_accuracies))
    
    print(f"Training mode: {training_mode}")
    print(f"Epochs completed: {num_epochs}")
    print(f"Train accuracies per epoch: {len(epoch_train_accuracies)}")
    print(f"Val accuracies per epoch: {len(epoch_val_accuracies)}")
    
    if test_acc is None or test_loss is None:
        print("\n⚠ WARNING: Test accuracy and/or test loss not provided!")
        print("   For assignment requirements, you must provide test results from click_test.parquet")
    
    # Create plots
    plot_training_results(
        epoch_train_accuracies=epoch_train_accuracies,
        epoch_train_losses=epoch_train_losses,
        epoch_val_accuracies=epoch_val_accuracies if len(epoch_val_accuracies) > 0 else None,
        epoch_val_losses=epoch_val_losses if len(epoch_val_losses) > 0 else None,
        test_acc=test_acc,
        test_loss=test_loss,
        save_dir=save_dir,
        suffix=suffix,
        step_train_losses=step_train_losses,
        step_train_accuracies=step_train_accuracies,
        step_numbers=step_numbers,
        best_val_acc=best_val_acc,
        training_mode=training_mode,
        num_epochs=num_epochs,
    )
    
    # Save metrics to JSON
    results_dir = Path(save_dir).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    metrics = {
        'training_dataset': 'click_train.parquet (preprocessed)',
        'test_dataset': 'click_test.parquet (preprocessed)',
        'epoch_train_accuracies': epoch_train_accuracies,
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_accuracies': epoch_val_accuracies,
        'epoch_val_losses': epoch_val_losses,
        'best_val_acc': best_val_acc,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'meets_40_percent_requirement': test_acc >= 0.40 if test_acc is not None else None,
        'training_mode': training_mode,
        'num_epochs': num_epochs,
    }
    
    metrics_file = results_dir / f'metrics{suffix}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training results from checkpoint\n'
                    'Training data: click_train.parquet\n'
                    'Test data: click_test.parquet\n'
                    'Expected test accuracy: ≥40%',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/best_model.pt)')
    parser.add_argument('--test_acc', type=float, required=True,
                       help='Test accuracy from click_test.parquet (REQUIRED)')
    parser.add_argument('--test_loss', type=float, required=True,
                       help='Test loss from click_test.parquet (REQUIRED)')
    parser.add_argument('--save_dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    
    args = parser.parse_args()
    
    load_checkpoint_and_plot(
        checkpoint_path=args.checkpoint_path,
        test_acc=args.test_acc,
        test_loss=args.test_loss,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
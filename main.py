from dataset import ParquetDataset, collate_queries
from train import Trainer
from plot_results import load_metrics_and_plot
from torch.utils.data import DataLoader
from pathlib import Path


def main(args):
    # Data paths
    train_data_path = 'data/preprocessed_click_train.parquet'
    test_data_path = 'data/preprocessed_click_test.parquet'
    val_data_path = 'data/preprocessed_click_val.parquet'

    print("="*60)
    print("CLICK PREDICTION TRAINING PIPELINE")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Training dataset: {train_data_path}")
    print(f"Validation dataset: {val_data_path}")
    print(f"Test dataset: {test_data_path}")
    print("="*60)

    # Create datasets
    train_dataset = ParquetDataset([train_data_path])
    val_dataset = ParquetDataset([val_data_path])
    test_dataset = ParquetDataset([test_data_path])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_queries,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_queries,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_queries,
        pin_memory=True,
        persistent_workers=True,
    )

    # Parse LoRA target modules if provided
    lora_target_modules = None
    if args.lora_target_modules:
        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
    
    # Create model-specific checkpoint directory
    # Sanitize model name for directory path
    model_dir_name = args.model_name.replace("/", "_").replace("\\", "_")
    checkpoint_dir = Path(args.checkpoint_dir) / model_dir_name
    
    # Initialize trainer
    trainer = Trainer(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_dir=str(checkpoint_dir),
        use_amp=args.use_amp,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules
    )
    
    # Run training
    print("\n" + "="*60)
    print("STEP 1: TRAINING")
    print("="*60)
    train_metrics = trainer.run_train(train_dataset, train_loader, val_loader=val_loader)
    
    # Test on test set
    print("\n" + "="*60)
    print("STEP 2: TESTING (click_test.parquet)")
    print("="*60)
    test_acc, test_loss = trainer.test(test_loader)
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Meets 40% requirement: {'YES ✓' if test_acc >= 0.40 else 'NO ⚠'}")
    
    # Generate plots
    print("\n" + "="*60)
    print("STEP 3: GENERATING PLOTS")
    print("="*60)
    load_metrics_and_plot(
        metrics_path=str(trainer.metrics_path),  # Path to training_metrics.json
        test_acc=test_acc,
        test_loss=test_loss,
        save_dir="results",
        model_name=args.model_name
    )
    
    # Final summary
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Best model: {trainer.best_model_path}")
    print(f"Last checkpoint: {trainer.last_checkpoint_path}")
    print(f"Training metrics: {trainer.metrics_path}")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test loss: {test_loss:.4f}")
    
    # Determine training mode subdirectory
    training_mode_dir = "lora" if args.use_lora else "full_finetuning"
    plots_dir = Path("results") / model_dir_name / training_mode_dir
    print(f"Plots saved to: {plots_dir}/")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Click Prediction Training Pipeline - Always trains and plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========================================
    # MODEL SELECTION
    # ========================================
    parser.add_argument("--model_name", type=str, default="roberta-base",
                       help="Model name to use (e.g., 'roberta-base', 'bert-base-uncased')")

    # ========================================
    # TRAINING HYPERPARAMETERS
    # ========================================
    parser.add_argument("--num_epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for training, validation, and testing")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                       help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                       help="Max gradient norm for gradient clipping")
    
    # ========================================
    # LOGGING AND CHECKPOINTING
    # ========================================
    parser.add_argument("--log_interval", type=int, default=50, 
                       help="Steps between log prints")
    parser.add_argument("--eval_interval", type=int, default=500, 
                       help="Steps between validation evaluations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                       help="Base directory to save model checkpoints")

    # ========================================
    # OPTIMIZATION FLAGS
    # ========================================
    parser.add_argument("--use_amp", action="store_true", 
                       help="Enable automatic mixed precision (FP16) training")
    parser.add_argument("--use_lora", action="store_true", 
                       help="Enable LoRA (Low-Rank Adaptation)")

    # ========================================
    # LORA-SPECIFIC ARGUMENTS
    # ========================================
    parser.add_argument("--lora_r", type=int, default=8, 
                       help="LoRA rank (dimensionality of low-rank matrices)")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                       help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, 
                       help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, default=None, 
                       help="Comma-separated list of module names to apply LoRA (e.g., 'query,value,key')")

    args = parser.parse_args()

    main(args)

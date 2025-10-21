from dataset import ParquetDataset, collate_queries
from train import Trainer
from plots import load_checkpoint_and_plot
from torch import DataLoader


def main(args):
    train_data_path = 'data/preprocessed_click_train.parquet'
    test_data_path = 'data/preprocessed_click_test.parquet'
    val_data_path = 'data/preprocessed_click_val.parquet'

    train_dataset = ParquetDataset([train_data_path])
    val_dataset = ParquetDataset([val_data_path])
    test_dataset = ParquetDataset([test_data_path])

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

    # Build test dataset/loader
    test_paths = ["preprocessed_click_test.parquet"]
    test_dataset = ParquetDataset(test_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_queries,
        pin_memory=True,
        persistent_workers=True,
    )

    load_checkpoint_and_plot(
        checkpoint_path=args.checkpoint_path,
        test_acc=args.test_acc,
        test_loss=args.test_loss,
        )


    trainer = Trainer(num_epochs=3, 
                 batch_size=16, learning_rate=2e-5, weight_decay=0.01,
                 warmup_steps=500, max_grad_norm=1.0, log_interval=50, 
                 eval_interval=500, checkpoint_dir="checkpoints", use_amp=True,
                 use_lora=False, use_qlora=False, lora_r=8, lora_alpha=16, 
                 lora_dropout=0.1, lora_target_modules=None)
    train_metrics = trainer.run_training(train_dataset, train_loader, val_loader=val_loader)
    test_acc, test_loss = trainer.test(test_loader)

    print(f"Final Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    load_checkpoint_and_plot(
        checkpoint_path=f"{trainer.checkpoint_dir}/best_model.pt",
        test_acc=test_acc,
        test_loss=test_loss,
        save_dir = "results"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Click Prediction Training and Evaluation")
    parser.add_argument("--train", action="store_true", help="Flag to trigger training")
    parser.add_argument("--plot", action="store_true", help="Flag to trigger plotting")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to model checkpoint for plotting")
    parser.add_argument("--test_acc", type=float, default=None, help="Test accuracy for plotting")
    parser.add_argument("--test_loss", type=float, default=None, help="Test loss for plotting")

    # Training/optimization args
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm (clipping)")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between log prints")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between eval runs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    # Boolean flags
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training (AMP)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA")

    # LoRA-specific args
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated module names for LoRA")

    args = parser.parse_args()

    main(args)
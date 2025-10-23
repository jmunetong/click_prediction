# Click Prediction Training Pipeline

A complete training pipeline for click prediction using transformer-based models. This pipeline trains models to predict which product a user will click from a set of search results.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Available Models](#available-models)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Single Training Run](#single-training-run)
  - [Training Parameters](#training-parameters)
- [Experiments](#experiments)
- [Output](#output)
- [Project Structure](#project-structure)

## Overview

This pipeline implements a cross-encoder approach for click prediction, where each (query, product) pair is encoded together and scored. The model learns to predict which product the user clicked from a candidate set.

**Key Features:**
- Multiple pooling strategies (CLS, mean pooling, attention pooling)
- Full fine-tuning with FP32 and FP16 (mixed precision)
- Automatic training, validation, and testing
- Comprehensive metrics tracking and visualization
- Checkpointing and model persistence

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM recommended
- See `requirements.txt` for Python dependencies

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd click_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your data files are in the `data/` directory:
   - `data/click_train.parquet` - Training queries with labels
   - `data/click_test.parquet` - Test queries with labels
   - `data/catalog.parquet` - Product catalog with descriptions

## Data Preparation

Before training, preprocess the raw data files:

```bash
python preprocess_data.py
```

This script:
- Tokenizes query-product pairs using RoBERTa tokenizer
- Creates train/validation split (90/10 by default)
- Outputs preprocessed files:
  - `data/preprocessed_click_train.parquet` - Training set
  - `data/preprocessed_click_val.parquet` - Validation set
  - `data/preprocessed_click_test.parquet` - Test set

**Configuration:**
- Maximum sequence length: 256 tokens
- Validation split: 10%
- Random seed: 42 (reproducible splits)

## Available Models

The pipeline supports three model architectures with different pooling strategies:

| Model Name | Architecture | Pooling Strategy | Description |
|------------|--------------|------------------|-------------|
| `cross_encoder` | RoBERTa-base | CLS token | Uses the [CLS] token representation |
| `cross_encoder_mean_pooling` | RoBERTa-base | Mean pooling | Averages all token embeddings |
| `cross_encoder_attention` | RoBERTa-base | Attention pooling | Learned attention weights over tokens (recommended) |

All models use `roberta-base` as the backbone transformer (125M parameters).

## Usage

### Quick Start

**Important:** Before running experiments, you must first preprocess the data:

```bash
python preprocess_data.py
```

This creates the tokenized datasets required for training. Then run all experiments:

```bash
bash run_experiment.sh
```

This will train two configurations (FP32 and FP16) and save results to separate directories.

**Note:** The preprocessed files (`preprocessed_click_train.parquet`, `preprocessed_click_val.parquet`, `preprocessed_click_test.parquet`) are stored in the `data/` directory and are required before training.

### Single Training Run

Train a single model with custom parameters:

```bash
python main.py \
    --model_name cross_encoder_attention \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --use_amp
```

**Example Outputs:**
- Trains for 5 epochs
- Tests on `preprocessed_click_test.parquet`
- Saves checkpoints to `checkpoints/cross_encoder_attention/`
- Generates plots in `results/cross_encoder_attention/full_finetuning/`

### Training Parameters

#### Model Selection
```bash
--model_name MODEL_NAME
```
Choose from: `cross_encoder`, `cross_encoder_mean_pooling`, `cross_encoder_attention` (default: `cross_encoder`)

#### Training Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | 3 | Number of training epochs |
| `--batch_size` | 16 | Batch size for training/validation/testing |
| `--learning_rate` | 2e-5 | Learning rate |
| `--weight_decay` | 0.01 | L2 regularization weight decay |
| `--warmup_steps` | 500 | Learning rate warmup steps |
| `--max_grad_norm` | 1.0 | Maximum gradient norm for clipping |

#### Logging and Checkpointing
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--log_interval` | 50 | Steps between progress logs |
| `--eval_interval` | 500 | Steps between validation evaluations |
| `--checkpoint_dir` | checkpoints | Base directory for model checkpoints |

#### Optimization Flags
| Parameter | Description |
|-----------|-------------|
| `--use_amp` | Enable FP16 mixed precision training (faster, lower memory) |

**Note:** `--use_amp` is automatically disabled on non-CUDA devices.

## Experiments

The `run_experiment.sh` script automates multiple training configurations:

### Run All Experiments
```bash
bash run_experiment.sh
```
Runs: FP32 full fine-tuning, FP16 full fine-tuning, and LoRA fine-tuning (Ignore LoRA for now. I was having issues with gradient tractability. For the purpose of time, I skipped this option)

### Run Specific Experiment
```bash
# FP32 full fine-tuning only
bash run_experiment.sh --only-fp32

# FP16 full fine-tuning only
bash run_experiment.sh --only-fp16
```

### Use Different Model
```bash
bash run_experiment.sh --model cross_encoder_attention
bash run_experiment.sh --model cross_encoder_mean_pooling --only-fp16
```

### Experiment Configuration

Default settings in `run_experiment.sh`:
```bash
NUM_EPOCHS=7
BATCH_SIZE=16
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
MAX_GRAD_NORM=1.0
LOG_INTERVAL=50
EVAL_INTERVAL=500
```

## Output

### Checkpoints
Saved to `checkpoints/{model_name}/`:
- `best_model.pt` - Model with best validation accuracy
- `last_model.pt` - Final model after training
- `training_metrics.json` - Complete training history
- `training_metrics.parquet` - Metrics in Parquet format
- `training.log` - Full training logs (when using run_experiment.sh)

### Plots
Saved to `results/{model_name}/full_finetuning/`:

1. **training_test_accuracy.png** - Primary plot showing training, validation, and test accuracy
2. **training_test_loss.png** - Loss curves for training, validation, and test
3. **combined_plot.png** - Side-by-side loss and accuracy plots
4. **step_level_plot.png** - Detailed step-by-step training progress
5. **plot_metrics.json** - Complete metrics including test results

### Console Output

The pipeline provides detailed progress information:

```
============================================================
CLICK PREDICTION TRAINING PIPELINE
============================================================
Model: cross_encoder_attention
Training dataset: data/preprocessed_click_train.parquet
Validation dataset: data/preprocessed_click_val.parquet
Test dataset: data/preprocessed_click_test.parquet
============================================================

============================================================
STEP 1: TRAINING
============================================================
[Training progress with loss and accuracy per epoch]

============================================================
STEP 2: TESTING (click_test.parquet)
============================================================
Final Test Accuracy: 0.4523 (45.23%)
Final Test Loss: 1.2341
Meets 40% requirement: YES ✓

============================================================
STEP 3: GENERATING PLOTS
============================================================
[Plot generation and saving]

============================================================
✓ PIPELINE COMPLETE!
============================================================
```

## Project Structure

```
click_prediction/
├── main.py                  # Main training script
├── preprocess_data.py       # Data preprocessing script
├── train.py                 # Trainer class implementation
├── model.py                 # Model architectures
├── dataset.py               # PyTorch dataset implementation
├── plot_results.py          # Plotting utilities
├── run_experiment.sh        # Experiment automation script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Data directory
│   ├── click_train.parquet
│   ├── click_test.parquet
│   ├── catalog.parquet
│   ├── preprocessed_click_train.parquet
│   ├── preprocessed_click_val.parquet
│   └── preprocessed_click_test.parquet
├── checkpoints/            # Model checkpoints
│   └── {model_name}/
│       ├── best_model.pt
│       ├── last_model.pt
│       └── training_metrics.json
└── results/                # Training plots
    └── {model_name}/
        └── full_finetuning/
            ├── training_test_accuracy.png
            ├── training_test_loss.png
            ├── combined_plot.png
            ├── step_level_plot.png
            └── plot_metrics.json
```






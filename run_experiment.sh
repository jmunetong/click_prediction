#!/bin/bash

# experiments.sh
# Runs training experiments: Full Fine-tuning (FP32 & FP16) and LoRA
# Each experiment trains and tests automatically

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_EPOCHS=1
BATCH_SIZE=16
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
MAX_GRAD_NORM=1.0
LOG_INTERVAL=50
EVAL_INTERVAL=500
MODEL_NAME="cross_encoder"

# LoRA Configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="query,value"

# Directories
CHECKPOINT_BASE_DIR="checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

run_full_finetuning_fp32() {
    print_header "EXPERIMENT 1: Full Fine-tuning (FP32 - Standard Precision)"
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/full_fp32"
    mkdir -p "$CHECKPOINT_DIR"

    print_info "Training with standard FP32 precision..."
    python main.py \
        --model_name $MODEL_NAME\
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --warmup_steps $WARMUP_STEPS \
        --max_grad_norm $MAX_GRAD_NORM \
        --log_interval $LOG_INTERVAL \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        2>&1 | tee "${CHECKPOINT_DIR}/training.log"

    if [ $? -eq 0 ]; then
        print_success "Experiment 1 completed successfully"
    else
        print_error "Experiment 1 failed"
    fi
}

run_full_finetuning_fp16() {
    print_header "EXPERIMENT 2: Full Fine-tuning (FP16 - Mixed Precision)"
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/full_fp16"
    mkdir -p "$CHECKPOINT_DIR"

    print_info "Training with FP16 mixed precision (AMP)..."
    python main.py \
        --model_name $MODEL_NAME\
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --warmup_steps $WARMUP_STEPS \
        --max_grad_norm $MAX_GRAD_NORM \
        --log_interval $LOG_INTERVAL \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --use_amp \
        2>&1 | tee "${CHECKPOINT_DIR}/training.log"

    if [ $? -eq 0 ]; then
        print_success "Experiment 2 completed successfully"
    else
        print_error "Experiment 2 failed"
    fi
}

run_lora() {
    print_header "EXPERIMENT 3: LoRA Fine-tuning"
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/lora"
    mkdir -p "$CHECKPOINT_DIR"

    print_info "Training with LoRA (Low-Rank Adaptation)..."
    python main.py \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --warmup_steps $WARMUP_STEPS \
        --max_grad_norm $MAX_GRAD_NORM \
        --log_interval $LOG_INTERVAL \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --use_amp \
        --use_lora \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_target_modules "$LORA_TARGET_MODULES" \
        2>&1 | tee "${CHECKPOINT_DIR}/training.log"

    if [ $? -eq 0 ]; then
        print_success "Experiment 3 completed successfully"
    else
        print_error "Experiment 3 failed"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================


if [ "$1" = "--only-fp16" ]; then
    run_full_finetuning_fp16
elif [ "$1" = "--only-lora" ]; then
    run_lora
else
    run_full_finetuning_fp32
fi
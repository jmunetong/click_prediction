#!/bin/bash

# experiments.sh
# Runs training experiments: Full Fine-tuning (FP32 & FP16) and LoRA
# Each experiment trains and tests automatically
# Usage:
#   ./experiments.sh [options]
#   Options:
#     --model <name>      Model name: cross_encoder, cross_encoder_mean_pooling, or cross_encoder_attention (default: cross_encoder)
#     --only-fp32         Run only FP32 experiment
#     --only-fp16         Run only FP16 experiment
#     --only-lora         Run only LoRA experiment
#   Examples:
#     ./experiments.sh --model cross_encoder_mean_pooling --only-fp16
#     ./experiments.sh --model cross_encoder_attention

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_EPOCHS=7
BATCH_SIZE=16
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
MAX_GRAD_NORM=1.0
LOG_INTERVAL=50
EVAL_INTERVAL=500
MODEL_NAME="cross_encoder"  # Default model

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
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

EXPERIMENT_MODE="all"  # Default: run all experiments

show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model <name>      Model name (default: cross_encoder)"
    echo "                      Options: cross_encoder, cross_encoder_mean_pooling, cross_encoder_attention"
    echo "  --only-fp32         Run only FP32 full fine-tuning experiment"
    echo "  --only-fp16         Run only FP16 full fine-tuning experiment"
    echo "  --only-lora         Run only LoRA fine-tuning experiment"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Run all experiments with default model"
    echo "  $0 --model cross_encoder_mean_pooling                # Run all experiments with mean pooling model"
    echo "  $0 --model cross_encoder_attention --only-fp16       # Run only FP16 with attention model"
    echo ""
}

while [ $# -gt 0 ]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --only-fp32)
            EXPERIMENT_MODE="fp32"
            shift
            ;;
        --only-fp16)
            EXPERIMENT_MODE="fp16"
            shift
            ;;
        --only-lora)
            EXPERIMENT_MODE="lora"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate model name
case "$MODEL_NAME" in
    cross_encoder|cross_encoder_mean_pooling|cross_encoder_attention)
        # Valid model name
        ;;
    *)
        echo "Error: Invalid model name '$MODEL_NAME'"
        echo "Valid options: cross_encoder cross_encoder_mean_pooling cross_encoder_attention"
        exit 1
        ;;
esac

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
    print_info "Model: $MODEL_NAME"
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${MODEL_NAME}/full_fp32"
    mkdir -p "$CHECKPOINT_DIR"

    print_info "Training with standard FP32 precision..."
    python main.py \
        --model_name "$MODEL_NAME" \
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
        print_success "Experiment 1 (FP32) completed successfully"
    else
        print_error "Experiment 1 (FP32) failed"
        return 1
    fi
}

run_full_finetuning_fp16() {
    print_header "EXPERIMENT 2: Full Fine-tuning (FP16 - Mixed Precision)"
    print_info "Model: $MODEL_NAME"
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${MODEL_NAME}/full_fp16"
    mkdir -p "$CHECKPOINT_DIR"

    print_info "Training with FP16 mixed precision (AMP)..."
    python main.py \
        --model_name "$MODEL_NAME" \
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
        print_success "Experiment 2 (FP16) completed successfully"
    else
        print_error "Experiment 2 (FP16) failed"
        return 1
    fi
}

run_lora() {
    print_header "EXPERIMENT 3: LoRA Fine-tuning"
    print_info "Model: $MODEL_NAME"
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${MODEL_NAME}/lora"
    mkdir -p "$CHECKPOINT_DIR"

    print_info "Training with LoRA (Low-Rank Adaptation)..."
    python main.py \
        --model_name "$MODEL_NAME" \
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
        print_success "Experiment 3 (LoRA) completed successfully"
    else
        print_error "Experiment 3 (LoRA) failed"
        return 1
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print_header "STARTING EXPERIMENTS"
print_info "Model: $MODEL_NAME"
print_info "Mode: $EXPERIMENT_MODE"
echo ""

case $EXPERIMENT_MODE in
    fp32)
        run_full_finetuning_fp32
        ;;
    fp16)
        run_full_finetuning_fp16
        ;;
    lora)
        run_lora
        ;;
    all)
        run_full_finetuning_fp32
        echo ""
        run_full_finetuning_fp16
        echo ""
        run_lora
        ;;
esac

print_header "ALL EXPERIMENTS COMPLETED"
print_success "Check the checkpoint directories for results:"
print_info "  ${CHECKPOINT_BASE_DIR}/${MODEL_NAME}/"
echo ""

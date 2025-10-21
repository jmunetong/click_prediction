#!/bin/bash

# experiments.sh
# Runs all training experiments: Full Fine-tuning (FP32 & FP16), LoRA, and QLoRA
# Each experiment trains, tests, and generates plots

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
MAX_GRAD_NORM=1.0
LOG_INTERVAL=50
EVAL_INTERVAL=500

# LoRA/QLoRA Configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="query,value"

# Directories
CHECKPOINT_BASE_DIR="checkpoints"
RESULTS_BASE_DIR="results"

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

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

run_full_finetuning_fp32() {
    print_header "EXPERIMENT 1: Full Fine-tuning (FP32 - Standard Precision)"
    
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/full_fp32"
    mkdir -p "$CHECKPOINT_DIR"
    
    print_info "Training with standard FP32 precision..."
    print_info "Checkpoint directory: $CHECKPOINT_DIR"
    
    python main.py --train --plot \
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
        print_info "Results saved to: $CHECKPOINT_DIR"
    else
        print_error "Experiment 1 failed"
        return 1
    fi
}

run_full_finetuning_fp16() {
    print_header "EXPERIMENT 2: Full Fine-tuning (FP16 - Mixed Precision)"
    
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/full_fp16"
    mkdir -p "$CHECKPOINT_DIR"
    
    print_info "Training with FP16 mixed precision (AMP)..."
    print_info "Checkpoint directory: $CHECKPOINT_DIR"
    
    python main.py --train --plot \
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
        print_info "Results saved to: $CHECKPOINT_DIR"
    else
        print_error "Experiment 2 failed"
        return 1
    fi
}

run_lora() {
    print_header "EXPERIMENT 3: LoRA Fine-tuning"
    
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/lora"
    mkdir -p "$CHECKPOINT_DIR"
    
    print_info "Training with LoRA (Low-Rank Adaptation)..."
    print_info "LoRA parameters: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
    print_info "Target modules: $LORA_TARGET_MODULES"
    print_info "Checkpoint directory: $CHECKPOINT_DIR"
    
    python main.py --train --plot \
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
        print_info "Results saved to: $CHECKPOINT_DIR"
    else
        print_error "Experiment 3 failed"
        return 1
    fi
}

run_qlora() {
    print_header "EXPERIMENT 4: QLoRA Fine-tuning (4-bit Quantization)"
    
    CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/qlora"
    mkdir -p "$CHECKPOINT_DIR"
    
    print_info "Training with QLoRA (4-bit Quantized LoRA)..."
    print_info "LoRA parameters: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
    print_info "Target modules: $LORA_TARGET_MODULES"
    print_info "Checkpoint directory: $CHECKPOINT_DIR"
    print_warning "Note: QLoRA requires CUDA. Mixed precision (AMP) is disabled for QLoRA."
    
    python main.py --train --plot \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --warmup_steps $WARMUP_STEPS \
        --max_grad_norm $MAX_GRAD_NORM \
        --log_interval $LOG_INTERVAL \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --use_qlora \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_target_modules "$LORA_TARGET_MODULES" \
        2>&1 | tee "${CHECKPOINT_DIR}/training.log"
    
    if [ $? -eq 0 ]; then
        print_success "Experiment 4 completed successfully"
        print_info "Results saved to: $CHECKPOINT_DIR"
    else
        print_error "Experiment 4 failed"
        return 1
    fi
}

# ============================================================================
# COMPARISON AND SUMMARY
# ============================================================================

generate_summary() {
    print_header "GENERATING EXPERIMENT SUMMARY"
    
    SUMMARY_FILE="${RESULTS_BASE_DIR}/experiment_summary.txt"
    mkdir -p "$RESULTS_BASE_DIR"
    
    echo "Experiment Summary" > "$SUMMARY_FILE"
    echo "==================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "Generated on: $(date)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Extract results from each experiment
    for exp_dir in "${CHECKPOINT_BASE_DIR}"/*; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            echo "Experiment: $exp_name" >> "$SUMMARY_FILE"
            echo "-------------------" >> "$SUMMARY_FILE"
            
            # Try to find best model checkpoint
            if [ -f "$exp_dir/best_model.pt" ] || [ -f "$exp_dir/best_model_lora.pt" ] || [ -f "$exp_dir/best_model_qlora.pt" ]; then
                print_success "Found results for: $exp_name"
                
                # Extract metrics from training log if available
                if [ -f "$exp_dir/training.log" ]; then
                    echo "Training completed successfully" >> "$SUMMARY_FILE"
                    
                    # Try to extract test accuracy from log
                    TEST_ACC=$(grep -oP "Final Test Accuracy: \K[0-9.]+" "$exp_dir/training.log" | tail -1)
                    TEST_LOSS=$(grep -oP "Final Test Loss: \K[0-9.]+" "$exp_dir/training.log" | tail -1)
                    
                    if [ -n "$TEST_ACC" ]; then
                        echo "Test Accuracy: $TEST_ACC" >> "$SUMMARY_FILE"
                        echo "Test Loss: $TEST_LOSS" >> "$SUMMARY_FILE"
                    fi
                fi
            else
                print_warning "No results found for: $exp_name"
                echo "Status: Incomplete or failed" >> "$SUMMARY_FILE"
            fi
            
            echo "" >> "$SUMMARY_FILE"
        fi
    done
    
    print_success "Summary generated: $SUMMARY_FILE"
    echo ""
    cat "$SUMMARY_FILE"
}

organize_results() {
    print_header "ORGANIZING RESULTS"
    
    # Create organized results directory structure
    ORGANIZED_DIR="${RESULTS_BASE_DIR}/organized"
    mkdir -p "$ORGANIZED_DIR"/{full_fp32,full_fp16,lora,qlora}
    
    # Copy plots to organized directory
    for exp_type in full_fp32 full_fp16 lora qlora; do
        if [ -d "${CHECKPOINT_BASE_DIR}/${exp_type}" ]; then
            # Copy plots
            if [ -d "results" ]; then
                cp results/*${exp_type}*.png "${ORGANIZED_DIR}/${exp_type}/" 2>/dev/null || true
                cp results/*.png "${ORGANIZED_DIR}/${exp_type}/" 2>/dev/null || true
            fi
            
            # Copy checkpoint
            cp -r "${CHECKPOINT_BASE_DIR}/${exp_type}/best_model"* "${ORGANIZED_DIR}/${exp_type}/" 2>/dev/null || true
            
            # Copy training log
            cp "${CHECKPOINT_BASE_DIR}/${exp_type}/training.log" "${ORGANIZED_DIR}/${exp_type}/" 2>/dev/null || true
            
            print_success "Organized results for: $exp_type"
        fi
    done
    
    print_success "Results organized in: $ORGANIZED_DIR"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    print_header "STARTING ALL EXPERIMENTS"
    print_info "Configuration:"
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Weight Decay: $WEIGHT_DECAY"
    echo "  Warmup Steps: $WARMUP_STEPS"
    echo ""
    
    # Track start time
    START_TIME=$(date +%s)
    
    # Track which experiments succeeded
    EXPERIMENTS_RUN=0
    EXPERIMENTS_SUCCEEDED=0
    
    # Run experiments
    experiments=(
        "run_full_finetuning_fp32"
        "run_full_finetuning_fp16"
        "run_lora"
        "run_qlora"
    )
    
    for exp_func in "${experiments[@]}"; do
        EXPERIMENTS_RUN=$((EXPERIMENTS_RUN + 1))
        
        if $exp_func; then
            EXPERIMENTS_SUCCEEDED=$((EXPERIMENTS_SUCCEEDED + 1))
        else
            print_error "Experiment failed: $exp_func"
            print_warning "Continuing with remaining experiments..."
        fi
        
        # Small delay between experiments
        sleep 2
    done
    
    # Generate summary
    generate_summary
    organize_results
    
    # Calculate total time
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    HOURS=$((TOTAL_TIME / 3600))
    MINUTES=$(((TOTAL_TIME % 3600) / 60))
    SECONDS=$((TOTAL_TIME % 60))
    
    # Final summary
    print_header "ALL EXPERIMENTS COMPLETED"
    echo ""
    echo "Experiments Run: $EXPERIMENTS_RUN"
    echo "Experiments Succeeded: $EXPERIMENTS_SUCCEEDED"
    echo "Experiments Failed: $((EXPERIMENTS_RUN - EXPERIMENTS_SUCCEEDED))"
    echo ""
    echo "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    print_info "Results Location:"
    echo "  Checkpoints: ${CHECKPOINT_BASE_DIR}/"
    echo "  Plots: results/"
    echo "  Organized: ${RESULTS_BASE_DIR}/organized/"
    echo "  Summary: ${RESULTS_BASE_DIR}/experiment_summary.txt"
    echo ""
    
    if [ $EXPERIMENTS_SUCCEEDED -eq $EXPERIMENTS_RUN ]; then
        print_success "All experiments completed successfully! 🎉"
        exit 0
    else
        print_warning "Some experiments failed. Check logs for details."
        exit 1
    fi
}

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

# Check if running with arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./experiments.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --dry-run           Show what would be run without executing"
    echo "  --skip-fp32         Skip FP32 full fine-tuning"
    echo "  --skip-fp16         Skip FP16 full fine-tuning"
    echo "  --skip-lora         Skip LoRA experiment"
    echo "  --skip-qlora        Skip QLoRA experiment"
    echo "  --only-fp32         Run only FP32 full fine-tuning"
    echo "  --only-fp16         Run only FP16 full fine-tuning"
    echo "  --only-lora         Run only LoRA experiment"
    echo "  --only-qlora        Run only QLoRA experiment"
    echo ""
    echo "Examples:"
    echo "  ./experiments.sh                    # Run all experiments"
    echo "  ./experiments.sh --only-lora        # Run only LoRA"
    echo "  ./experiments.sh --skip-qlora       # Run all except QLoRA"
    exit 0
fi

if [ "$1" == "--dry-run" ]; then
    print_info "DRY RUN MODE - Commands that would be executed:"
    echo ""
    print_info "1. Full Fine-tuning (FP32)"
    print_info "2. Full Fine-tuning (FP16)"
    print_info "3. LoRA Fine-tuning"
    print_info "4. QLoRA Fine-tuning"
    echo ""
    print_info "Each experiment will:"
    echo "  - Train the model"
    echo "  - Test on click_test.parquet"
    echo "  - Generate plots"
    echo "  - Save checkpoints"
    exit 0
fi

# Handle experiment selection
if [ "$1" == "--only-fp32" ]; then
    run_full_finetuning_fp32
elif [ "$1" == "--only-fp16" ]; then
    run_full_finetuning_fp16
elif [ "$1" == "--only-lora" ]; then
    run_lora
elif [ "$1" == "--only-qlora" ]; then
    run_qlora
else
    # Run all experiments (with optional skips)
    main
fi
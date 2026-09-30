#!/bin/bash
# Executes the 5-seed statistical validation protocol outlined in Response Letter

TEST_DIR="./data/darkface_test/degraded/"
GT_DIR="./data/darkface_test/clean/"

# 1. Run 5 independent seeds
for SEED in 42 123 456 789 999; do
    echo "========================================"
    echo "Running QFI-Diff Inference for Seed $SEED..."
    echo "========================================"
    OUT="./outputs/qfidiff_seed_$SEED/"
    
    python pipeline/inference_qfidiff.py \
        --input_dir $TEST_DIR \
        --output_dir $OUT \
        --seed $SEED \
        --vqgan_weights ./checkpoints/vqgan_finetuned_epoch50.pth
        
    python evaluation/evaluate_metrics.py \
        --pred_dir $OUT \
        --gt_dir $GT_DIR
done

# 2. Run Wilcoxon Signed-Rank Test against Baselines
echo "========================================"
echo "Running Statistical Significance Tests..."
echo "========================================"
python evaluation/statistical_tests.py \
    --qfi_scores ./outputs/qfidiff_seed_42/ssim_scores.json \
    --baseline_scores ./outputs/baselines/diffpir/ssim_scores.json ./outputs/baselines/resshift/ssim_scores.json

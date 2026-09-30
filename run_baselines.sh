#!/bin/bash
# Evaluates baselines using official off-the-shelf checkpoints as defined in Table 2.

TEST_DIR="./data/darkface_test/degraded/"
GT_DIR="./data/darkface_test/clean/"
OUTPUT_BASE="./outputs/baselines/"

echo "1. Running SwinIR (Public pretrained checkpoint, off-the-shelf)..."
git clone https://github.com/JingyunLiang/SwinIR.git
cd SwinIR
python main_test_swinir.py --task classical_sr --folder_lq ../$TEST_DIR --model_path ../checkpoints/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth
cd ..

echo "2. Running DiffPIR (Public pretrained checkpoint, off-the-shelf)..."
git clone https://github.com/yuanjiali/DiffPIR.git
cd DiffPIR
python main_diffpir.py --task deblur --test_dir ../$TEST_DIR --save_dir ../$OUTPUT_BASE/diffpir/
cd ..

echo "3. Running ResShift (Public pretrained checkpoint, off-the-shelf)..."
git clone https://github.com/zsyOAOA/ResShift.git
cd ResShift
python test_resshift.py --task real_sr --input_dir ../$TEST_DIR --output_dir ../$OUTPUT_BASE/resshift/
cd ..

echo "Evaluation complete. Use evaluation/evaluate_metrics.py to calculate PI, FID, etc."

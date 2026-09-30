import numpy as np
from scipy.stats import wilcoxon
import argparse
import ast

def holm_bonferroni(p_values, alpha=0.05):
    sorted_indices = np.argsort(p_values)
    m = len(p_values)
    reject = np.zeros(m, dtype=bool)
    
    for k, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (m - k)
        if p_values[idx] <= adjusted_alpha:
            reject[idx] = True
        else:
            break
    return reject

def run_tests(qfi_scores, baseline_scores):
    # Expecting arrays of per-image SSIM scores over 5 seeds
    qfi_arr = np.array(ast.literal_eval(qfi_scores))
    baseline_arr = np.array(ast.literal_eval(baseline_scores))
    
    stat, p_value = wilcoxon(qfi_arr, baseline_arr)
    print(f"Wilcoxon Signed-Rank Test p-value: {p_value:.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qfi_scores", type=str, required=True, help="List of scores")
    parser.add_argument("--baseline_scores", type=str, required=True)
    args = parser.parse_args()
    run_tests(args.qfi_scores, args.baseline_scores)

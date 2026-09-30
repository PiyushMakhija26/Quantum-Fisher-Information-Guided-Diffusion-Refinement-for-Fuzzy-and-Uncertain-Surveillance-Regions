import numpy as np
import json
from scipy.stats import wilcoxon
import argparse

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

def run_tests(qfi_json, baseline_jsons):
    with open(qfi_json, 'r') as f:
        qfi_arr = np.array(json.load(f))
        
    p_values = []
    names = []
    
    for b_json in baseline_jsons:
        with open(b_json, 'r') as f:
            b_arr = np.array(json.load(f))
        stat, p_value = wilcoxon(qfi_arr, b_arr)
        p_values.append(p_value)
        names.append(b_json)
        
    reject_flags = holm_bonferroni(p_values)
    
    print("Wilcoxon Signed-Rank Test (Holm-Bonferroni Corrected, alpha=0.05):")
    for name, p, reject in zip(names, p_values, reject_flags):
        print(f"vs {name} | p-value: {p:.4e} | Statistically Significant: {reject}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qfi_scores", type=str, required=True, help="Path to QFI SSIM JSON")
    parser.add_argument("--baseline_scores", nargs='+', required=True, help="Paths to Baseline SSIM JSONs")
    args = parser.parse_args()
    run_tests(args.qfi_scores, args.baseline_scores)

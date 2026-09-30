import os
import cv2
import torch
import numpy as np
import argparse
import json
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths
import pyiqa

def evaluate_metrics(pred_dir, gt_dir, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    iqa_ma = pyiqa.create_metric('ma').cuda()
    iqa_niqe = pyiqa.create_metric('niqe').cuda()
    
    psnr_scores, ssim_scores, lpips_scores, ma_scores, niqe_scores = [], [], [], [], []
    
    for filename in sorted(os.listdir(pred_dir)):
        if not filename.endswith(('.png', '.jpg')): continue
        
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        
        pred_img = cv2.imread(pred_path)
        gt_img = cv2.imread(gt_path)
        
        psnr_scores.append(psnr(gt_img, pred_img))
        ssim_scores.append(ssim(gt_img, pred_img, channel_axis=2))
        
        pred_tensor = (torch.tensor(pred_img).permute(2,0,1).unsqueeze(0).float()/127.5 - 1).cuda()
        gt_tensor = (torch.tensor(gt_img).permute(2,0,1).unsqueeze(0).float()/127.5 - 1).cuda()
        
        with torch.no_grad():
            lpips_scores.append(loss_fn_alex(pred_tensor, gt_tensor).item())
            ma_scores.append(iqa_ma(pred_tensor).item())
            niqe_scores.append(iqa_niqe(pred_tensor).item())

    # Save per-image arrays for statistical testing
    with open(os.path.join(pred_dir, "ssim_scores.json"), "w") as f:
        json.dump(ssim_scores, f)

    fid_score = calculate_fid_given_paths([gt_dir, pred_dir], batch_size=16, device='cuda', dims=2048)
    
    ma_mean = np.mean(ma_scores)
    niqe_mean = np.mean(niqe_scores)
    pi_score = 0.5 * ((10 - ma_mean) + niqe_mean)
    
    return {
        "PSNR": np.mean(psnr_scores),
        "SSIM": np.mean(ssim_scores),
        "LPIPS": np.mean(lpips_scores),
        "FID": fid_score,
        "Ma": ma_mean,
        "NIQE": niqe_mean,
        "PI": pi_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    args = parser.parse_args()
    
    res = evaluate_metrics(args.pred_dir, args.gt_dir)
    for k, v in res.items():
        print(f"{k}: {v:.4f}")

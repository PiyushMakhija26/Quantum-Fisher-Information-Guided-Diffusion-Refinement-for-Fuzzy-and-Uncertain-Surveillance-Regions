import torch
import cv2
import os
import argparse
from diffusers import StableDiffusionPipeline
from qfi_core.qfi_extractor import QFIMaskGenerator
from pipeline.ddim_sampler import qfi_fusion_step

def run_inference(args):
    print(f"Starting QFI-Diff Inference (Seed: {args.seed})...")
    torch.manual_seed(args.seed)
    
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(args.device)
    if args.vqgan_weights and os.path.exists(args.vqgan_weights):
        pipe.vae.load_state_dict(torch.load(args.vqgan_weights))
        print("Loaded finetuned VQ-GAN decoder weights.")

    qfi_gen = QFIMaskGenerator(k=4, J=1.0, h=0.5, device=args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for img_name in os.listdir(args.input_dir):
        if not img_name.endswith(('.png', '.jpg')): continue
        img_path = os.path.join(args.input_dir, img_name)
        
        img_cv = cv2.imread(img_path)
        img_tensor_gray = torch.tensor(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)).unsqueeze(0).unsqueeze(0).float().to(args.device) / 255.0
        
        # 1. Generate Deterministic Reliability Mask
        mask = qfi_gen.generate_mask(img_tensor_gray)
        
        # 2. Encode Observation
        img_tensor_rgb = torch.tensor(img_cv).permute(2,0,1).unsqueeze(0).float().to(args.device) / 127.5 - 1
        with torch.no_grad():
            z_0_obs = pipe.vae.encode(img_tensor_rgb).latent_dist.sample() * pipe.vae.config.scaling_factor
        
            # 3. Reverse Diffusion SDE with Manifold Consistency Constraint
            z_t = torch.randn_like(z_0_obs)
            prompt_embeds = pipe._encode_prompt("", args.device, 1, False)
            
            pipe.scheduler.set_timesteps(args.steps)
            for t in pipe.scheduler.timesteps:
                z_t = qfi_fusion_step(pipe, z_t, t, prompt_embeds, z_0_obs, mask)
                
            # 4. Decode to Pixel Space
            restored = pipe.vae.decode(z_t / pipe.vae.config.scaling_factor).sample
            
        restored = (restored / 2 + 0.5).clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy() * 255
        cv2.imwrite(os.path.join(args.output_dir, img_name), restored.astype('uint8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vqgan_weights", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    run_inference(parser.parse_args())

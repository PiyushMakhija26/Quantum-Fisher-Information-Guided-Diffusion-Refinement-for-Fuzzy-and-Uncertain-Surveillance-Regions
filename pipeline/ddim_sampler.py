import torch
import torch.nn.functional as F

def qfi_fusion_step(pipeline, noisy_latent_t, t, text_embeddings, z_0_obs, qfi_mask_highres):
    # Ensure mask has shape (1, 1, H, W) and matches latent spatial resolution
    if qfi_mask_highres.dim() == 2:
        mask_4d = qfi_mask_highres.unsqueeze(0).unsqueeze(0)
    elif qfi_mask_highres.dim() == 3:
        mask_4d = qfi_mask_highres.unsqueeze(1)
    else:
        mask_4d = qfi_mask_highres

    M_z = F.interpolate(
        mask_4d, 
        size=(noisy_latent_t.shape[2], noisy_latent_t.shape[3]), 
        mode='bilinear', 
        align_corners=False
    ).to(device=noisy_latent_t.device, dtype=noisy_latent_t.dtype)
    
    # UNet forward pass
    noise_pred = pipeline.unet(noisy_latent_t, t, encoder_hidden_states=text_embeddings).sample
    scheduler = pipeline.scheduler
    
    # Timestep indexing and cumulative alphas
    t_idx = t.item() if isinstance(t, torch.Tensor) else int(t)
    alpha_t = scheduler.alphas_cumprod[t_idx].to(device=noisy_latent_t.device, dtype=noisy_latent_t.dtype)
    
    step_size = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_t = t_idx - step_size
    if prev_t >= 0:
        alpha_t_prev = scheduler.alphas_cumprod[prev_t].to(device=noisy_latent_t.device, dtype=noisy_latent_t.dtype)
    else:
        alpha_t_prev = torch.tensor(1.0, device=noisy_latent_t.device, dtype=noisy_latent_t.dtype)
    
    # DDIM reverse step formulation
    pred_x0 = (noisy_latent_t - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
    dir_xt = torch.sqrt(1.0 - alpha_t_prev) * noise_pred
    z_t_prev_pred = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
    
    # Observation perturbation forward step
    noise = torch.randn_like(z_0_obs)
    z_y_prev = torch.sqrt(alpha_t_prev) * z_0_obs + torch.sqrt(1.0 - alpha_t_prev) * noise
    
    # Manifold consistency fusion
    z_t_prev_fused = (1.0 - M_z) * z_y_prev + M_z * z_t_prev_pred
    
    return z_t_prev_fused

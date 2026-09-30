import torch
import torch.nn.functional as F

def qfi_fusion_step(pipeline, noisy_latent_t, t, text_embeddings, z_0_obs, qfi_mask_highres):
    M_z = F.avg_pool2d(qfi_mask_highres.unsqueeze(0).unsqueeze(0), kernel_size=8).squeeze()
    
    noise_pred = pipeline.unet(noisy_latent_t, t, encoder_hidden_states=text_embeddings).sample
    scheduler = pipeline.scheduler
    
    alpha_t = scheduler.alphas_cumprod[t]
    step_size = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_t_prev = scheduler.alphas_cumprod[t - step_size] if t > step_size else torch.tensor(1.0)
    
    pred_x0 = (noisy_latent_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
    dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
    z_t_prev_pred = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
    
    noise = torch.randn_like(z_0_obs)
    z_y_prev = torch.sqrt(alpha_t_prev) * z_0_obs + torch.sqrt(1 - alpha_t_prev) * noise
    
    z_t_prev_fused = (1.0 - M_z) * z_y_prev + M_z * z_t_prev_pred
    
    return z_t_prev_fused

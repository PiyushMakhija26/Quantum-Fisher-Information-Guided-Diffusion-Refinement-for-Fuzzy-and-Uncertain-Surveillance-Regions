import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import os

class DarkFaceUnpairedDataset(Dataset):
    def __init__(self, data_dir, target_size=(512, 512)):
        self.img_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))])
        self.target_size = target_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.resize(img, self.target_size)
        return torch.tensor(img).permute(2,0,1).float() / 127.5 - 1

def train_vqgan(args):
    print(f"Initializing VQ-GAN Domain Adaptation on {args.device}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").to(args.device)
    
    vae.encoder.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    
    dataset = DarkFaceUnpairedDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = AdamW(vae.decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(args.epochs):
        vae.train()
        epoch_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(args.device)
            optimizer.zero_grad()
            
            latents = vae.encode(imgs).latent_dist.sample()
            restored_imgs = vae.decode(latents).sample
            
            loss = criterion(restored_imgs, imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/len(dataloader):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), f"{args.output_dir}/vqgan_finetuned_epoch{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Unpaired DarkFace images (5,000 subset)")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    train_vqgan(parser.parse_args())

"""
Enhanced training with architectural losses:
- Edge preservation loss
- Structural similarity loss
- Perceptual loss
- Total variation loss for smoothness
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.models import vgg16
from models_enhanced import EnhancedUNetGenerator, PatchDiscriminator
from dataset_enhanced import get_dataloader
from utils import save_sample
import yaml
import numpy as np

def load_config(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better texture preservation"""
    def __init__(self, device):
        super().__init__()
        try:
            vgg = vgg16(pretrained=True).features[:16].to(device).eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.criterion = nn.L1Loss()
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load VGG for perceptual loss: {e}")
            self.available = False
        
    def forward(self, pred, target):
        if not self.available:
            return torch.tensor(0.0, device=pred.device)
        
        # Convert single channel to 3 channels
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.criterion(pred_features, target_features)

class EdgeLoss(nn.Module):
    """Sobel-based edge detection loss for sharp walls"""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        self.sobel_y = sobel_y.view(1, 1, 3, 3)
        
    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # Edge detection
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
        
        return F.l1_loss(pred_edge, target_edge)

class TVLoss(nn.Module):
    """Total Variation loss for smoothness"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:-1]), 2).sum()
        
        return (h_tv + w_tv) / (batch_size * h_x * w_x)

def extract_edges_from_input(img):
    """Extract edges from input floorplan using Canny for cleaner architectural lines"""
    # Convert RGB to grayscale
    gray = 0.299 * img[:,0:1,:,:] + 0.587 * img[:,1:2,:,:] + 0.114 * img[:,2:3,:,:]
    
    # Use Canny edge detection (better for architectural lines than Sobel)
    gray_np = gray.squeeze(0).cpu().numpy()  # Shape: [B, 1, H, W] -> [B, H, W]
    edges_list = []
    
    for i in range(gray_np.shape[0]):
        # Convert to uint8 for OpenCV
        gray_uint8 = (gray_np[i] * 255).astype(np.uint8)
        # Apply Canny edge detection
        edge = cv2.Canny(gray_uint8, threshold1=100, threshold2=200)
        edges_list.append(edge)
    
    # Convert back to tensor
    edges = torch.from_numpy(np.array(edges_list)).unsqueeze(1).float() / 255.0
    edges = edges.to(img.device)
    
    return edges

def train(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = cfg.get("batch_size", 8)
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", 2e-4)
    lambda_l1 = cfg.get("lambda_l1", 100.0)
    img_size = cfg.get("img_size", 256)
    
    # Loss weights - INCREASED for better wall quality
    lambda_edge = cfg.get("lambda_edge", 100.0)  # ← Increased from 50.0
    lambda_perceptual = cfg.get("lambda_perceptual", 10.0)
    lambda_tv = cfg.get("lambda_tv", 5.0)  # ← Increased from 0.1
    
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Loss weights - L1: {lambda_l1}, Edge: {lambda_edge}, "
          f"Perceptual: {lambda_perceptual}, TV: {lambda_tv}")
    
    os.makedirs(cfg["checkpoints"], exist_ok=True)
    os.makedirs(cfg["outputs"], exist_ok=True)
    sample_dir = os.path.join(cfg["outputs"], "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Dataset
    print(f"\nLoading dataset from: {cfg['data']['floorplan_dir']}")
    dl = get_dataloader(
        cfg["data"]["floorplan_dir"], 
        cfg["data"].get("target_dir"), 
        img_size=img_size, 
        batch_size=batch_size,
        shuffle=True,
        augment=True
    )
    print(f"Dataset size: {len(dl.dataset)} images")
    print(f"Batches per epoch: {len(dl)}")
    
    # Models
    print("\nInitializing models...")
    netG = EnhancedUNetGenerator(in_channels=3, out_channels=1).to(device)
    netD = PatchDiscriminator(in_channels=4).to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in netG.parameters())
    d_params = sum(p.numel() for p in netD.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Losses
    criterionGAN = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    criterionEdge = EdgeLoss()
    criterionPerceptual = PerceptualLoss(device)
    criterionTV = TVLoss()
    
    # Optimizers with learning rate scheduling
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=tuple(cfg.get("betas", [0.5, 0.999])))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=tuple(cfg.get("betas", [0.5, 0.999])))
    
    # Learning rate schedulers
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=epochs)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=epochs)
    
    global_step = 0
    best_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        netG.train()
        netD.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_edge_loss = 0.0
        
        for batch_idx, batch in enumerate(dl):
            real_A = batch["floor"].to(device)  # [B,3,H,W]
            real_B = batch["target"].to(device)  # [B,1,H,W]
            bs = real_A.size(0)
            
            # Ground truth labels
            valid = torch.ones((bs, 1, 30, 30), device=device)
            fake_label = torch.zeros_like(valid)
            
            # Extract edges from input floorplan
            input_edges = extract_edges_from_input(real_A)
            
            # ---------------------
            # Train Generator
            # ---------------------
            optG.zero_grad()
            
            fake_B, pred_edges = netG(real_A)  # Get both heightmap and edge predictions
            
            # Scale to [0,1]
            fake_B_norm = (fake_B + 1) / 2
            real_B_norm = real_B
            
            # GAN loss
            pred_fake = netD(real_A, fake_B_norm)
            g_gan = criterionGAN(pred_fake, valid)
            
            # L1 loss
            g_l1 = criterionL1(fake_B_norm, real_B_norm) * lambda_l1
            
            # Edge loss (preserve sharp walls)
            g_edge = criterionEdge(fake_B_norm, real_B_norm) * lambda_edge
            
            # Edge prediction loss (match input edges)
            # RESIZE input_edges to match pred_edges size
            input_edges_resized = F.interpolate(input_edges, size=pred_edges.shape[2:], mode='bilinear', align_corners=False)
            g_edge_pred = F.binary_cross_entropy(pred_edges, input_edges_resized) * lambda_edge
            
            # Perceptual loss
            if criterionPerceptual.available:
                g_perceptual = criterionPerceptual(fake_B_norm, real_B_norm) * lambda_perceptual
            else:
                g_perceptual = torch.tensor(0.0, device=device)
            
            # Total variation loss (smoothness in non-edge areas)
            g_tv = criterionTV(fake_B_norm) * lambda_tv
            
            # Combined generator loss
            g_loss = g_gan + g_l1 + g_edge + g_edge_pred + g_perceptual + g_tv
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optG.step()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optD.zero_grad()
            
            pred_real = netD(real_A, real_B_norm)
            loss_real = criterionGAN(pred_real, valid)
            
            pred_fake_detach = netD(real_A, fake_B_norm.detach())
            loss_fake = criterionGAN(pred_fake_detach, fake_label)
            
            d_loss = (loss_real + loss_fake) * 0.5
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optD.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_l1_loss += g_l1.item()
            epoch_edge_loss += g_edge.item()
            
            # Logging
            if global_step % 100 == 0:
                perc_val = g_perceptual.item() if isinstance(g_perceptual, torch.Tensor) else 0.0
                print(f"[Epoch {epoch}/{epochs}][{batch_idx}/{len(dl)}] "
                      f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} "
                      f"(L1: {g_l1.item():.3f}, Edge: {g_edge.item():.3f}, "
                      f"Perc: {perc_val:.3f})")
            
            if global_step % 500 == 0:
                save_sample(real_A, real_B_norm, fake_B_norm, sample_dir, step=global_step)
            
            global_step += 1
        
        # Learning rate scheduling
        schedulerG.step()
        schedulerD.step()
        
        # Epoch summary
        avg_g_loss = epoch_g_loss / len(dl)
        avg_d_loss = epoch_d_loss / len(dl)
        avg_l1_loss = epoch_l1_loss / len(dl)
        avg_edge_loss = epoch_edge_loss / len(dl)
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} Complete - Time: {epoch_time:.1f}s")
        print(f"Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f}")
        print(f"Avg L1: {avg_l1_loss:.4f} | Avg Edge: {avg_edge_loss:.4f}")
        print(f"Learning Rate: {schedulerG.get_last_lr()[0]:.6f}")
        print(f"{'='*60}\n")
        
        # Save checkpoints
        save_checkpoint = False
        checkpoint_name = None
        
        if epoch % 5 == 0:
            save_checkpoint = True
            checkpoint_name = f"epoch_{epoch}"
        
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            save_checkpoint = True
            checkpoint_name = "best"
            print(f"🎯 New best model! Loss: {best_loss:.4f}")
        
        if save_checkpoint:
            ckpt_g = os.path.join(cfg["checkpoints"], f"netG_{checkpoint_name}.pth")
            ckpt_d = os.path.join(cfg["checkpoints"], f"netD_{checkpoint_name}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optG.state_dict(),
                'loss': avg_g_loss,
            }, ckpt_g)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optD.state_dict(),
            }, ckpt_d)
            
            print(f"💾 Saved checkpoints: {checkpoint_name}")
    
    # Final save
    final_path = os.path.join(cfg["checkpoints"], "netG_final.pth")
    torch.save(netG.state_dict(), final_path)
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"💾 Final model saved: {final_path}")
    print(f"🏆 Best model saved: checkpoints/netG_best.pth")
    print(f"📊 Training samples: {sample_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train enhanced floorplan-to-3D model")
    parser.add_argument("--cfg", default="config_enhanced.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Add OpenCV import for Canny edge detection
    global cv2
    try:
        import cv2
    except ImportError:
        print("⚠️ Warning: OpenCV not installed. Using Sobel edge detection instead.")
        # Define a dummy function to avoid crash
        def extract_edges_from_input(img):
            # Fall back to Sobel if no OpenCV
            gray = 0.299 * img[:,0:1,:,:] + 0.587 * img[:,1:2,:,:] + 0.114 * img[:,2:3,:,:]
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device)
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            edge_x = F.conv2d(gray, sobel_x, padding=1)
            edge_y = F.conv2d(gray, sobel_y, padding=1)
            edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
            edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
            return edges
    
    train(cfg_path=args.cfg)
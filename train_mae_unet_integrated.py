# Integrated MAE-UNet Training Script for Cryo-EM Image Segmentation
# Combines MAE pre-training and MAE-UNet training in sequence
# Author: CryoSegNet Team
# Date: 2024

from utils.accuracy import dice_score, jaccard_score
from dataset.dataset import CryoEMDataset
from dataset.augmented_dataset import AugmentedCryoEMDataset
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loss import DiceLoss
import glob
from tqdm import tqdm
import time
import os
from torch.nn.parallel import DataParallel
import argparse
from torch.cuda.amp import autocast, GradScaler

def load_pretrained_mae_encoder(mae_checkpoint_path, device):
    """
    Load pretrained MAE encoder for feature extraction
    Exit program if loading fails
    """
    import sys
    sys.path.append('Cryo-EMMAE')
    from models.mae_model import MaskedAutoencoderViT
    import yaml
    
    if not os.path.exists(mae_checkpoint_path):
        print(f"[ERROR] MAE checkpoint file does not exist: {mae_checkpoint_path}")
        print("[ERROR] Please check the path or use test_mae_features.py for testing")
        sys.exit(1)
    
    try:
        config_path = os.path.join(os.path.dirname(mae_checkpoint_path), '..', 'params', '45.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            img_size = config.get('img_size', 64)
            patch_size = config.get('patch_size', 4)
            embed_dim = config.get('embed_dim', 192)
            depth = config.get('depth', 14)
            num_heads = config.get('num_heads', 1)
        else:
            img_size, patch_size, embed_dim, depth, num_heads = 64, 4, 192, 14, 1
        
        mae_model = MaskedAutoencoderViT(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=1,
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads,
            decoder_embed_dim=128, 
            decoder_depth=7, 
            decoder_num_heads=8,
            mlp_ratio=2.0
        )
        
        checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('decoder') and not k.startswith('mask_token'):
                encoder_state_dict[k] = v
        
        mae_model.load_state_dict(encoder_state_dict, strict=False)
        mae_model = mae_model.to(device)
        mae_model.eval()
        
        for param in mae_model.parameters():
            param.requires_grad = False
        
        print(f"[INFO] Pretrained MAE encoder loaded: {mae_checkpoint_path}")
        print(f"[INFO] MAE config: img_size={img_size}, patch_size={patch_size}, embed_dim={embed_dim}")
        
        return mae_model, embed_dim
        
    except Exception as e:
        print(f"[ERROR] MAE model loading failed: {e}")
        print("[ERROR] Please use test_mae_features.py for detailed testing")
        sys.exit(1)

def extract_mae_features(mae_model, images, device, feature_channels=8):
    """
    Extract features using pretrained MAE model and reconstruct as extra channels
    Exit program if numerical anomalies occur during feature extraction
    Args:
        mae_model: Pretrained MAE encoder
        images: [B, 1, 1024, 1024] input images
        device: Device
        feature_channels: Number of feature channels to extract
    Returns:
        enhanced_images: [B, 1+feature_channels, 1024, 1024] enhanced input
    """
    import sys
    
    batch_size = images.shape[0]
    feature_maps = []
    
    for b in range(batch_size):
        single_image = images[b:b+1]  # [1, 1, 1024, 1024]
        
        tiles = []
        for gy in range(16):  # 16x16 grid
            for gx in range(16):
                y0, x0 = gy * 64, gx * 64
                tile = single_image[:, :, y0:y0+64, x0:x0+64]  # [1, 1, 64, 64]
                tiles.append(tile)
        
        tiles_tensor = torch.cat(tiles, dim=0)  # [256, 1, 64, 64]
        
        tile_batch_size = 64
        paved_tiles = []
        
        with torch.no_grad():
            for start in range(0, tiles_tensor.shape[0], tile_batch_size):
                end = min(start + tile_batch_size, tiles_tensor.shape[0])
                tile_batch = tiles_tensor[start:end].to(device)
                
                try:
                    tokens = mae_model.infer_latent(tile_batch)  # [B_t, 257, D]
                    tokens = tokens[:, 1:, :]  # [B_t, 256, D]
                    tokens = tokens[:, :, :feature_channels]  # [B_t, 256, C]
                    
                    if torch.isnan(tokens).any():
                        print(f"[ERROR] NaN detected in MAE feature extraction (batch {b+1})")
                        print("[ERROR] Please use test_mae_features.py for detailed testing")
                        sys.exit(1)
                    
                    if torch.isinf(tokens).any():
                        print(f"[ERROR] Inf detected in MAE feature extraction (batch {b+1})")
                        print("[ERROR] Please use test_mae_features.py for detailed testing")
                        sys.exit(1)
                    
                    bt = tokens.shape[0]
                    tokens_spatial = tokens.reshape(bt, 16, 16, feature_channels).permute(0, 3, 1, 2).contiguous()  # [B_t, C, 16, 16]
                    tokens_ups = F.interpolate(tokens_spatial, scale_factor=4, mode='bilinear', align_corners=False)  # upsample to 64x64
                    
                    if torch.isnan(tokens_ups).any() or torch.isinf(tokens_ups).any():
                        print(f"[ERROR] Anomalies detected in upsampled features (batch {b+1})")
                        print("[ERROR] Please use test_mae_features.py for detailed testing")
                        sys.exit(1)
                    
                    paved_tiles.append(tokens_ups)
                    
                except Exception as e:
                    print(f"[ERROR] MAE feature extraction failed (batch {b+1}): {e}")
                    print("[ERROR] Please use test_mae_features.py for detailed testing")
                    sys.exit(1)

        paved_tiles = torch.cat(paved_tiles, dim=0)  # [256, C, 64, 64]
        paved_tiles = paved_tiles.view(16, 16, feature_channels, 64, 64).permute(2, 0, 3, 1, 4).contiguous()  # [16,16,C,64,64] → [C,1024,1024]
        feature_full = paved_tiles.view(feature_channels, 1024, 1024)
        
        if torch.isnan(feature_full).any():
            print(f"[ERROR] NaN detected in final feature map (batch {b+1})")
            print("[ERROR] Please use test_mae_features.py for detailed testing")
            sys.exit(1)
        
        if torch.isinf(feature_full).any():
            print(f"[ERROR] Inf detected in final feature map (batch {b+1})")
            print("[ERROR] Please use test_mae_features.py for detailed testing")
            sys.exit(1)
        
        feature_maps.append(feature_full)
    
    feature_maps = torch.stack(feature_maps, dim=0)  # [B, feature_channels, 1024, 1024]
    feature_maps = feature_maps.to(images.device)
    
    enhanced_images = torch.cat([images, feature_maps], dim=1)  # [B, 1+feature_channels, 1024, 1024]
    
    if torch.isnan(enhanced_images).any():
        print("[ERROR] NaN detected in final enhanced images")
        print("[ERROR] Please use test_mae_features.py for detailed testing")
        sys.exit(1)
    
    if torch.isinf(enhanced_images).any():
        print("[ERROR] Inf detected in final enhanced images")
        print("[ERROR] Please use test_mae_features.py for detailed testing")
        sys.exit(1)
    
    return enhanced_images

def mae_unet_training_phase(args, device, mae_checkpoint_path):
    """
    Phase 2: MAE-UNet Training with Pretrained MAE Features
    """
    print("=" * 60)
    print("PHASE 2: MAE-UNet Training with Pretrained MAE Features")
    print("=" * 60)
    
    mae_encoder, embed_dim = load_pretrained_mae_encoder(mae_checkpoint_path, device)
    
    # Recursively gather train/val image paths across all sub-datasets
    train_image_path = []
    val_image_path = []
    train_image_path.extend(glob.glob(os.path.join(args.train_dataset_path, '**', 'train', 'images', '*.jpg'), recursive=True))
    val_image_path.extend(glob.glob(os.path.join(args.train_dataset_path, '**', 'val', 'images', '*.jpg'), recursive=True))
    
    if len(train_image_path) == 0 or len(val_image_path) == 0:
        print(f"[ERROR] No train/val images found recursively under {args.train_dataset_path}")
        print("[HINT] Ensure structure like */train/images/*.jpg and */val/images/*.jpg exists")
        return None, None, None
    
    if args.use_augmentation:
        print("[INFO] Using augmented training set to reduce overfitting")
        train_ds = AugmentedCryoEMDataset(img_dir=train_image_path, transform=None, augment=True)
    else:
        print("[INFO] Using standard training set")
        train_ds = CryoEMDataset(img_dir=train_image_path, transform=None)
    
    val_ds = CryoEMDataset(img_dir=val_image_path, transform=None)
    
    print(f"[INFO] Found {len(train_ds)} examples in the training set across all subsets...")
    print(f"[INFO] Found {len(val_ds)} examples in the validation set across all subsets...")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        shuffle=True, 
        batch_size=args.batch_size, 
        pin_memory=args.pin_memory, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, 
        shuffle=True, 
        batch_size=args.batch_size, 
        pin_memory=args.pin_memory, 
        num_workers=args.num_workers
    )
    print(f"[INFO] Train Loader Length {len(train_loader)}...")
    
    if args.use_dropout:
        from models.model_5_layers_mae_aug_dropout import UNETWithExtraChannelsDropout
        model = UNETWithExtraChannelsDropout(in_channels=1 + args.feature_channels, dropout_rate=args.dropout_rate)
        print(f"[INFO] Using Dropout regularization, dropout_rate={args.dropout_rate}")
    else:
        from models.model_5_layers_mae_aug import UNETWithExtraChannels
        model = UNETWithExtraChannels(in_channels=1 + args.feature_channels)
        print("[INFO] Using standard UNet model")
    
    if args.use_dataparallel and torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = DataParallel(model)
    
    model = model.to(device)
    
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion1 = BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion2 = DiceLoss()
    
    bce_weight = args.bce_weight
    dice_weight = args.dice_weight
    
    print(f"[INFO] Using class-balanced BCE loss, pos_weight={pos_weight.item()}")
    print(f"[INFO] Loss function weights: BCE={bce_weight}, Dice={dice_weight}")
    
    # Use AdamW for better convergence with transformer-based models
    if args.optimizer == 'adamw':
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    else:
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.01
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_step_size, 
            gamma=args.lr_gamma
        )
    else:
        scheduler = None
    
    # Calculate steps per epoch
    train_steps = len(train_ds) // args.batch_size
    val_steps = len(val_ds) // args.batch_size
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[INFO] Number of Training Steps : {train_steps}")
    print(f"[INFO] Number of Validation Steps : {val_steps}")
    print(f"[INFO] Total Number of Parameters : {total_params:,}")
    
    # Initialize training history
    H = {
        "train_loss": [], "val_loss": [], 
        "train_dice_score": [], "val_dice_score": [], 
        "train_jaccard_score": [], "val_jaccard_score": [], 
        "epochs": []
    }
    
    best_val_loss = float("inf")
    best_val_dice = 0.0
    patience_counter = 0
    patience = 15
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    print("[INFO] Training UNet with pretrained MAE features...")
    start_time = time.time()
    
    for e in tqdm(range(args.num_epochs)):
        model.train()
        
        train_loss = 0
        train_dice_scores = []
        train_jaccard_scores = []
        
        # Training loop
        for i, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            
            enhanced_x = extract_mae_features(mae_encoder, x, device, args.feature_channels)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=args.use_amp):
                pred = model(enhanced_x)  # [B, 1, 1024, 1024]
                loss1 = criterion1(pred, y)  # BCEWithLogits expects logits
                loss2 = criterion2(torch.sigmoid(pred), y)
                loss = bce_weight * loss1 + dice_weight * loss2
            
            scaler.scale(loss).backward()
            
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            probs = torch.sigmoid(pred.detach())
            pred_binary = (probs > args.dice_threshold).float()
            y_binary = (y > 0.5).float()
            
            train_dice_scores.append(dice_score(y_binary, pred_binary).item())
            train_jaccard_scores.append(jaccard_score(y_binary, pred_binary).item())
            
            if i % 200 == 0:
                pred_min, pred_max = pred.min().item(), pred.max().item()
                prob_min, prob_max = probs.min().item(), probs.max().item()
                y_mean, prob_mean = y.mean().item(), probs.mean().item()
                
                y_binary_sum = y_binary.sum().item()
                pred_binary_sum = pred_binary.sum().item()
                intersection = (pred_binary * y_binary).sum().item()
                union = y_binary_sum + pred_binary_sum - intersection
                current_dice = (2.0 * intersection + 1e-6) / (union + intersection + 1e-6) if union > 0 else 0.0
                
                prob_above_threshold = (probs > args.dice_threshold).float().mean().item()
                
                print(f"[DEBUG] Batch {i}: pred_range=[{pred_min:.3f}, {pred_max:.3f}], "
                      f"prob_range=[{prob_min:.3f}, {prob_max:.3f}], "
                      f"target_mean={y_mean:.3f}, prob_mean={prob_mean:.3f}")
                print(f"[DEBUG] Batch {i}: y_binary_sum={y_binary_sum:.0f}, pred_binary_sum={pred_binary_sum:.0f}, "
                      f"intersection={intersection:.0f}, dice={current_dice:.4f}, "
                      f"prob_above_threshold={prob_above_threshold:.3f}")
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_dice_score = np.mean(train_dice_scores)
        train_jaccard_score = np.mean(train_jaccard_scores)
        
        # Validation loop
        val_loss = 0    
        val_dice_scores = [] 
        val_jaccard_scores = []
        
        model.eval()
        with torch.no_grad(): 
            for i, data in enumerate(val_loader):
                x, y = data
                x, y = x.to(device), y.to(device)
                
                enhanced_x = extract_mae_features(mae_encoder, x, device, args.feature_channels)
                
                with autocast(enabled=args.use_amp):
                    pred = model(enhanced_x)  # [B, 1, 1024, 1024]
                    loss1 = criterion1(pred, y)
                    loss2 = criterion2(torch.sigmoid(pred), y)
                    loss = bce_weight * loss1 + dice_weight * loss2
                val_loss += loss.item()
                
                val_probs = torch.sigmoid(pred.detach())
                val_pred_binary = (val_probs > args.dice_threshold).float()
                val_y_binary = (y > 0.5).float()
                
                val_dice_scores.append(dice_score(val_y_binary, val_pred_binary).item())
                val_jaccard_scores.append(jaccard_score(val_y_binary, val_pred_binary).item())
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_dice_score = np.mean(val_dice_scores)
        val_jaccard_score = np.mean(val_jaccard_scores)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Update training history
        H["train_loss"].append(train_loss)
        H["val_loss"].append(val_loss)
        H["train_dice_score"].append(train_dice_score)
        H["train_jaccard_score"].append(train_jaccard_score)
        H["val_dice_score"].append(val_dice_score)
        H["val_jaccard_score"].append(val_jaccard_score)
        H["epochs"].append(e + 1)
        
        # Print training information
        current_lr = optimizer.param_groups[0]['lr']
        print("[INFO] EPOCH: {}/{}".format(e + 1, args.num_epochs))
        print("Train Loss: {:.4f}, Validation Loss: {:.4f}".format(train_loss, val_loss))
        print("Train Dice: {:.4f}, Validation Dice: {:.4f}".format(train_dice_score, val_dice_score))
        print("Learning Rate: {:.6f}".format(current_lr))
        print("-" * 60)
        
        if val_dice_score > best_val_dice:
            best_val_dice = val_dice_score
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = os.path.join(
                args.output_path, 
                "models", 
                "mae_unet_best_val_dice.pth"
            )
            
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            
            print(f"[INFO] New best model saved with validation Dice: {val_dice_score:.4f}")
        else:
            patience_counter += 1
            
        dice_gap = train_dice_score - val_dice_score
        if dice_gap > 0.15:
            print(f"[WARNING] Possible overfitting! Train Dice({train_dice_score:.3f}) - Val Dice({val_dice_score:.3f}) = {dice_gap:.3f}")
            
        if patience_counter >= patience:
            print(f"[INFO] Early stopping triggered! No improvement for {patience} epochs.")
            print(f"[INFO] Best validation Dice: {best_val_dice:.4f}")
            break
        
        # Save checkpoint periodically
        if (e + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_path, 
                "models", 
                f'mae_unet_checkpoint_epoch_{e+1}.pth'
            )
            
            checkpoint = {
                'epoch': e + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'training_history': H
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"[INFO] Checkpoint saved at epoch {e+1}")
    
    # Training completed
    end_time = time.time()
    total_time = end_time - start_time
    
    print("[INFO] MAE-UNet training completed!")
    print(f"[INFO] Total time taken: {total_time/3600:.2f} hours")
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(
        args.output_path, 
        "models", 
        "mae_unet_final.pth"
    )
    
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    
    print(f"[INFO] Final model saved to {final_model_path}")
    
    return H, best_val_loss, final_model_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Integrated MAE-UNet Training for Cryo-EM Segmentation')
    
    # Data paths
    parser.add_argument('--train_dataset_path', type=str, default='train_dataset/',
                       help='Root path containing one or multiple subsets with */train/* and */val/*')
    parser.add_argument('--output_path', type=str, default='output/mae_unet_integrated/',
                       help='Output directory for models and logs')
    
    parser.add_argument('--mae_config', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='MAE encoder configuration')
    parser.add_argument('--img_size', type=int, default=64,
                       help='ViT input tile size (use 64 for 64x64 tiles; upstream images stay 1024x1024)')
    parser.add_argument('--patch_size', type=int, default=4,
                       help='ViT patch size inside each 64x64 tile (4 -> 16x16 tokens per tile)')
    parser.add_argument('--adapter_type', type=str, default='standard',
                       choices=['standard', 'lightweight'],
                       help='Type of pyramid adapter')
    parser.add_argument('--freeze_mae', action='store_true',
                       help='Freeze MAE encoder during training')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for MAE-UNet training (increased for better convergence)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of MAE-UNet training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for MAE-UNet training (kept lower for fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for MAE-UNet training (aligned with Cryo-EMMAE)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=50,
                       help='Step size for step scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                       help='Gamma for step scheduler')
    parser.add_argument('--save_interval', type=int, default=20,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--use_dataparallel', action='store_true',
                       help='Enable torch.nn.DataParallel for multi-GPU (default off to save memory)')
    parser.add_argument('--tile_batch_size', type=int, default=64,
                       help='Per-step tile batch size (tiles per forward); lower to reduce peak memory')
    # MAE feature extraction options
    parser.add_argument('--use_pretrained_mae', action='store_true',
                       help='Use pretrained MAE model to extract features for UNet training')
    parser.add_argument('--mae_checkpoint_path', type=str, 
                       default='Cryo-EMMAE/checkpoints/45/MAE_epoch_500.pth.tar',
                       help='Path to pretrained MAE checkpoint')
    parser.add_argument('--feature_channels', type=int, default=8,
                       help='Number of MAE feature channels to extract and concatenate')
    
    parser.add_argument('--use_amp', action='store_true',
                       help='Enable mixed precision training for memory saving')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                       help='Gradient accumulation steps to simulate larger batch')
    
    # Data augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Enable data augmentation to reduce overfitting')
    
    # Dropout regularization
    parser.add_argument('--use_dropout', action='store_true',
                       help='Enable Dropout regularization in UNet model')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate for regularization (default: 0.1)')
    
    # Loss function configuration
    parser.add_argument('--pos_weight', type=float, default=4.0,
                       help='Positive class weight for BCE loss (default: 4.0)')
    parser.add_argument('--bce_weight', type=float, default=0.5,
                       help='Weight for BCE loss in combined loss (default: 0.5)')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                       help='Weight for Dice loss in combined loss (default: 0.5)')
    parser.add_argument('--dice_threshold', type=float, default=0.5,
                       help='Threshold for binary prediction in Dice calculation (default: 0.5)')
    
    # Gradient clipping
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable, default: 1.0)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--pin_memory', action='store_true',
                       help='Enable pin memory for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'models'), exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("Integrated MAE-UNet Training Configuration")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"{key:30}: {value}")
    print("=" * 60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    print("\n" + "=" * 60)
    print("Using original train/val split")
    print("=" * 60)
    print("Training will use the original train/val split in the dataset")
    print("Data path:", args.train_dataset_path)
    
    if args.use_pretrained_mae:
        print("\n" + "=" * 60)
        print("Starting MAE-UNet training")
        print("=" * 60)
        
        history, best_val_loss, final_model_path = mae_unet_training_phase(args, device, args.mae_checkpoint_path)
        if final_model_path is not None:
            print(f"\n[INFO] Training completed with best validation loss: {best_val_loss:.4f}")
            print(f"[INFO] Final model saved to: {final_model_path}")
            
            print("\nTraining completed with original dataset split!")
        else:
            print("\n[ERROR] Training failed")
    else:
        print("[ERROR] Current configuration requires pretrained MAE model (--use_pretrained_mae)")
        print("[ERROR] Please set --use_pretrained_mae and provide correct --mae_checkpoint_path")
    
    print("[INFO] Training completed!")

if __name__ == '__main__':
    main()

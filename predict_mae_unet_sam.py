# Integrated MAE-UNet + SAM Prediction Script for Cryo-EM Image Segmentation
# Uses trained MAE-UNet model and SAM for particle detection
# Outputs particle coordinates and result images
# Author: CryoSegNet Team
# Date: 2024

import copy
from utils.denoise import denoise, denoise_jpg_image
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
import numpy as np
import torch
import cv2
import glob
import os
from dataset.dataset import transform
from models.model_5_layers_mae_aug import UNETWithExtraChannels
from models.model_5_layers_mae_aug_dropout import UNETWithExtraChannelsDropout
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import statistics as st
import argparse
import csv
from datetime import datetime
from tqdm import tqdm

DEFAULT_COORD_DIR = os.path.join('output', 'coordinates')
DEFAULT_RESULT_DIR = os.path.join('output', 'results')

def load_unet_with_extra(model_path, in_channels=9, device='cuda'):
    print(f"[INFO] Loading UNet model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    has_dropout = any('conv.4.weight' in key and len(state_dict[key].shape) == 4 for key in state_dict.keys())
    
    if has_dropout:
        print("[INFO] Detected Dropout model, using UNETWithExtraChannelsDropout")
        model = UNETWithExtraChannelsDropout(in_channels=in_channels, dropout_rate=0.1)
    else:
        print("[INFO] Detected standard model, using UNETWithExtraChannels")
        model = UNETWithExtraChannels(in_channels=in_channels)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def load_pretrained_mae_encoder(mae_checkpoint_path, device='cuda'):
    import sys, os
    sys.path.append('Cryo-EMMAE')
    from models.mae_model import MaskedAutoencoderViT
    import yaml
    cfg_path = os.path.join(os.path.dirname(mae_checkpoint_path), '..', 'params', '45.yaml')
    img_size = 64; patch_size = 4; embed_dim = 192; depth = 14; num_heads = 1
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            y = yaml.safe_load(f)
        img_size = y.get('img_size', img_size)
        patch_size = y.get('patch_size', patch_size)
        embed_dim = y.get('embed_dim', embed_dim)
        depth = y.get('depth', depth)
        num_heads = y.get('num_heads', num_heads)
    mae = MaskedAutoencoderViT(
        img_size=img_size, patch_size=patch_size, in_chans=1,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        decoder_embed_dim=128, decoder_depth=7, decoder_num_heads=8,
        mlp_ratio=2.0
    )
    ckpt = torch.load(mae_checkpoint_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    enc_state = {k: v for k, v in state.items() if not k.startswith('decoder') and 'mask_token' not in k}
    mae.load_state_dict(enc_state, strict=False)
    mae = mae.to(device)
    mae.eval()
    for p in mae.parameters():
        p.requires_grad = False
    return mae

@torch.no_grad()
def extract_mae_paved_features(mae_model, image_1024, feature_channels=8, device='cuda'):
    # image_1024: torch.Tensor [1,1,1024,1024]
    tiles = []
    for gy in range(16):
        for gx in range(16):
            y0, x0 = gy*64, gx*64
            tiles.append(image_1024[:, :, y0:y0+64, x0:x0+64])
    tiles = torch.cat(tiles, dim=0).to(device)  # [256,1,64,64]
    paved = []
    bs = 64
    for s in range(0, tiles.shape[0], bs):
        e = min(s+bs, tiles.shape[0])
        tb = tiles[s:e]
        tokens = mae_model.infer_latent(tb)  # [B_t,257,D]
        tokens = tokens[:, 1:, :][:, :, :feature_channels]  # [B_t,256,C]
        bt = tokens.shape[0]
        tok_sp = tokens.reshape(bt, 16, 16, feature_channels).permute(0,3,1,2).contiguous()  # [B_t,C,16,16]
        tok_up = torch.nn.functional.interpolate(tok_sp, scale_factor=4, mode='bilinear', align_corners=False)  # [B_t,C,64,64]
        paved.append(tok_up)
    paved = torch.cat(paved, dim=0)  # [256,C,64,64]
    paved = paved.view(16,16,feature_channels,64,64).permute(2,0,3,1,4).contiguous()
    feats_full = paved.view(feature_channels,1024,1024).unsqueeze(0)  # [1,C,1024,1024]
    return feats_full

def load_sam_model(model_type='vit_h', sam_checkpoint=None, device='cuda'):
    if sam_checkpoint is None:
        sam_checkpoint = config.sam_checkpoint
    print(f"[INFO] Loading SAM model: {model_type}")
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )
    return sam_model, mask_generator

def get_annotations(anns):
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img

def prepare_plot(image, predicted_mask, sam_mask, coords, image_path, args):
    fig = plt.figure(figsize=(40, 30))
    try:
        plt.subplot(221)
        plt.title('Testing Image', fontsize=14)
        plt.imshow(image, cmap='gray')
        
        plt.subplot(222)
        plt.title('MAE-UNet Mask', fontsize=14)
        plt.imshow(predicted_mask, cmap='gray')
        
        plt.subplot(223)
        plt.title('SAM Mask', fontsize=14)
        if sam_mask is not None:
            plt.imshow(sam_mask, cmap='gray')
        else:
            plt.imshow(np.zeros_like(image), cmap='gray')
            plt.text(0.5, 0.5, 'No SAM masks', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.subplot(224)
        plt.title('Final Picked Particles', fontsize=14)
        plt.imshow(coords, cmap='gray')
        
        path = image_path.split("/")[-1]
        path = path.replace(".png", "_result.png")
        path = path.replace(".jpg", "_result.jpg")
        path = path.replace(".mrc", "_result.jpg")
        
        output_dir = getattr(args, 'output_path', DEFAULT_RESULT_DIR)
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        final_path = os.path.join(results_dir, f'{path}')
        
        plt.savefig(final_path.replace('.jpg', '_plot.jpg'), bbox_inches='tight', dpi=100)
        cv2.imwrite(final_path, coords)
    finally:
        plt.close(fig)
        plt.clf()

def make_predictions(unet_model, sam_mask_generator, image_path, args, mae_model=None):
    lower_path = image_path.lower()
    if lower_path.endswith('.jpg') or lower_path.endswith('.png'):
        image = cv2.imread(image_path, 0)
    elif lower_path.endswith('.mrc'):
        image = denoise(image_path)
    else:
        print(f"[WARNING] Unsupported image format: {image_path}")
        return None
    orig_height, orig_width = image.shape
    orig_image = copy.deepcopy(image)
    if image.shape[0] != 1024 or image.shape[1] != 1024:
        image_1024 = cv2.resize(image, (1024, 1024))
    else:
        image_1024 = image
    segment_mask = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2BGR)

    img_t = torch.from_numpy(image_1024).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1,1,1024,1024]
    img_t = img_t.to(args.device)
    feats = extract_mae_paved_features(mae_model, img_t, feature_channels=args.feature_channels, device=args.device)  # [1,C,1024,1024]
    x = torch.cat([img_t, feats], dim=1)  # [1,1+C,1024,1024]
    with torch.no_grad():
        pred = unet_model(x)
        predicted_mask = torch.sigmoid(pred).cpu().numpy().squeeze(0).squeeze(0)
    sam_input = np.repeat(transform(predicted_mask)[:,:,None], 3, axis=-1)
    predicted_mask_resized = cv2.resize(predicted_mask, (orig_width, orig_height))
    try:
        masks = sam_mask_generator.generate(sam_input)
        sam_mask = get_annotations(masks)
        if sam_mask is not None:
            sam_mask = cv2.resize(sam_mask, (orig_width, orig_height))
        else:
            sam_mask = np.zeros((orig_height, orig_width, 4))
    except Exception as e:
        print(f"[WARNING] SAM mask generation failed: {e}")
        masks = []
        sam_mask = np.zeros((orig_height, orig_width, 4))
    bboxes = []
    confidences = []
    for i in range(0, len(masks)):
        if masks[i]["predicted_iou"] > args.sam_iou_thresh:
            box = masks[i]["bbox"]
            bboxes.append(box)
            confidences.append(masks[i]["predicted_iou"])
    particles = []
    if len(bboxes) >= 1:
        x_ = st.mode([box[2] for box in bboxes])
        y_ = st.mode([box[3] for box in bboxes])
        d_ = np.sqrt((x_ * orig_width / 1024)**2 + (y_ * orig_height / 1024)**2)
        r_ = int(d_//2)

        th = r_ * 0.2
        
        for i, b in enumerate(bboxes):
            if b[2] < x_ + th and b[2] > x_ - th/3 and b[3] < y_ + th and b[3] > y_ - th/3:
                x_new = int((b[0] + b[2]/2) / 1024 * orig_width)
                y_new = int((b[1] + b[3]/2) / 1024 * orig_height)
                confidence = confidences[i]
                particles.append({
                    "x": x_new,
                    "y": y_new,
                    "radius": r_,
                    "confidence": float(confidence)
                })
                cv2.putText(segment_mask, f"{confidence:.2f}", (x_new, y_new), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                coords = cv2.circle(segment_mask, (x_new, y_new), r_, (0, 0, 255), 8)
        if not getattr(args, 'no_images', False):
            try:
                prepare_plot(orig_image, predicted_mask_resized, sam_mask, coords, image_path, args)
            except Exception as e:
                print(f"[WARNING] Failed to save result plot: {e}")
        
    output_dir = getattr(args, 'output_path', DEFAULT_COORD_DIR)
    coords_dir = os.path.join(output_dir, "coordinates")
    os.makedirs(coords_dir, exist_ok=True)
    
    csv_filename = os.path.basename(image_path).replace(".jpg", ".csv").replace(".png", ".csv").replace(".mrc", ".csv")
    csv_path = os.path.join(coords_dir, csv_filename)
    
    if particles:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Radius", "Confidence"])
            for p in particles:
                writer.writerow([p["x"], p["y"], p["radius"], p["confidence"]])
        print(f"[INFO] Saved {len(particles)} particles to {csv_path}")
        
        if not getattr(args, 'no_images', False):
            results_dir = os.path.join(output_dir, "results")
            print(f"[INFO] Result image saved to {results_dir}")
    else:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Radius", "Confidence"])
        print(f"[INFO] No particles detected in {os.path.basename(image_path)}, created empty CSV")
    return particles

def main():
    parser = argparse.ArgumentParser(description='MAE-UNet + SAM Prediction for Cryo-EM Images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained UNetWithExtraChannels model')
    parser.add_argument('--mae_config', type=str, default='small', choices=['small', 'base', 'large'], help='MAE configuration used during training')
    parser.add_argument('--img_size', type=int, default=64, help='ViT tile size inside MAE (fixed by pretrain, usually 64)')
    parser.add_argument('--patch_size', type=int, default=4, help='ViT patch size inside MAE tile (e.g., 4)')
    parser.add_argument('--feature_channels', type=int, default=8, help='Number of MAE feature channels to concatenate (w)')
    parser.add_argument('--mae_checkpoint_path', type=str, required=True, help='Path to pretrained MAE checkpoint for features')
    # backward-compat arg (no-op)
    parser.add_argument('--pred_tile_bs', type=int, default=64, help='Deprecated: unused in UNetWithExtraChannels pipeline')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'], help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='Path to SAM checkpoint (uses config default if not specified)')
    parser.add_argument('--sam_iou_thresh', type=float, default=0.94, help='SAM IoU threshold for mask filtering')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_path', type=str, default='output', help='Output directory for results and coordinates')
    parser.add_argument('--file_pattern', type=str, default='auto', help='File pattern for batch processing; use auto to include common image/MRC formats')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--no_images', action='store_true', help='Skip image output, only save coordinates (faster)')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output path: {args.output_path}")
    print(f"[INFO] Image output: {'Disabled (coordinates only)' if args.no_images else 'Enabled'}")
    unet_model = load_unet_with_extra(args.model_path, in_channels=1+args.feature_channels, device=device)
    mae_model = load_pretrained_mae_encoder(args.mae_checkpoint_path, device=device)
    _, sam_mask_generator = load_sam_model(args.sam_model_type, args.sam_checkpoint, device)
    if os.path.isfile(args.input_path):
        input_paths = [args.input_path]
    elif os.path.isdir(args.input_path):
        if args.file_pattern.lower() == 'auto':
            candidate_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.mrc']
        else:
            candidate_patterns = [args.file_pattern]
        input_paths = []
        for pattern in candidate_patterns:
            input_paths.extend(glob.glob(os.path.join(args.input_path, pattern)))
        input_paths = sorted(set(input_paths))
        if not input_paths:
            print(f"[ERROR] No files found matching pattern(s): {candidate_patterns}")
            return
    else:
        print(f"[ERROR] Input path {args.input_path} does not exist")
        return
    print(f"[INFO] Processing {len(input_paths)} file(s)")
    
    os.makedirs(args.output_path, exist_ok=True)
    coords_dir = os.path.join(args.output_path, "coordinates")
    os.makedirs(coords_dir, exist_ok=True)
    
    if not args.no_images:
        results_dir = os.path.join(args.output_path, "results")
        os.makedirs(results_dir, exist_ok=True)
    
    total_particles = 0
    processed_files = 0
    for image_path in tqdm(input_paths, desc="Processing images"):
        try:
            particles = make_predictions(unet_model, sam_mask_generator, image_path, args, mae_model)
            if particles:
                total_particles += len(particles)
                processed_files += 1
                print(f"[INFO] {image_path}: {len(particles)} particles detected")
            else:
                print(f"[INFO] {image_path}: No particles detected")
        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")
            continue
    print("=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Processed files: {processed_files}/{len(input_paths)}")
    print(f"Total particles detected: {total_particles}")
    print(f"Coordinates saved to: {os.path.join(args.output_path, 'coordinates')}")
    if not args.no_images:
        print(f"Result images saved to: {os.path.join(args.output_path, 'results')}")
    else:
        print("Result images: Skipped (--no_images enabled)")
    print("=" * 60)

if __name__ == '__main__':
    main()

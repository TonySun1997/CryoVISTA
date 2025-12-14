# CryoVISTA
CryoVISTA is an innovative particle picking algorithm for cryo-electron microscopy (cryo-EM). It relies on two key design improvements:

1. **MAE-based feature fusion**  
A pre-trained Masked Autoencoder pulls high-level semantic features. These features are spatially remapped, then combined with raw micrographs to form a dual-channel input that highlights discriminative details.

2. **Hierarchical U-Net+SAM segmentation cascade**  
An attention-gated U-Net generates rough particle masks. These masks guide the Segment Anything Model (SAM) to output precise, sub-pixel particle localizations.

## Quick Start

### 1. Clone the Repository
Download the project and navigate to the project directory:
```bash
git clone https://github.com/TonySun1997/CryoVISTA.git
cd CryoVISTA
```
### 2. Download the model
To run CryoVISTA smoothly, you need to download **five folders** (MAE_pretrain, output, pretrained_models, test_dataset and train_dataset) from the Google Drive link https://drive.google.com/drive/folders/1SrwFJkXzNk1ggwUBeLfqHE3YsfFdcPju?usp=drive_link. After downloading, place all five folders directly in the **root directory of the CryoVISTA project** (i.e., under `CryoVISTA/`).




### 3. Create and Activate Conda Environment
Create the dedicated conda environment using the provided environment.yml file, then activate it:
```
# Create the conda environment
conda env create -f environment.yml

# Activate the cryovista environment
conda activate cryovista
```

### Step 4: Run Particle Picking
Execute the following command to start cryo-EM particle picking. This command uses **8 extra feature channels** and disables image output (optimizes speed/storage efficiency):

```bash
python predict_mae_unet_sam.py \
  --model_path output/mae_unet_integrated/models/8_extra_channel.pth \
  --mae_checkpoint_path MAE_pretrain/MAE_epoch_500.pth.tar \
  --input_path test_dataset \
  --output_path output/prediction_results \
  --img_size 64 --patch_size 4 \
  --feature_channels 8 \
  --sam_model_type vit_h \
  --pred_tile_bs 64 \
  --device cuda \
  --no_images
  ```
| Parameter | Description |
|-----------|-------------|
| --model_path | Path to the pre-trained MAE-U-Net integrated model (8 extra channels version) |
| --mae_checkpoint_path | Path to the MAE pre-trained checkpoint file (ensure this file is placed in `MAE_pretrain/` as instructed earlier) |
| --input_path | Directory of test micrographs (cryo-EM datasets) |
| --output_path | Directory to save particle picking results |
| --img_size 64 --patch_size 4 | Image size and patch size for feature extraction |
| --feature_channels 8 | Use 8 extra feature channels (core setting for this version) |
| --sam_model_type vit_h | SAM model type (vit_h = high-performance version) |
| --pred_tile_bs 64 | Batch size for tiled prediction (balances speed and memory) |
| --device cuda | Use GPU (CUDA) for acceleration (replace with `cpu` if no GPU available) |
| --no_images | Disable image output (only save particle coordinate, reduces storage usage) |


### Step 5: Train the Model on Custom Datasets
Use the command below to train the CryoVISTA model on your custom cryo-EM datasets. We provide two protein datasets as demos for quick testing:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mae_unet_integrated.py \
  --train_dataset_path train_dataset \
  --output_path output/mae_unet_integrated \
  --num_epochs 100 --batch_size 8 \
  --learning_rate 0.0003 \
  --weight_decay 5e-3 \
  --img_size 64 --patch_size 4 \
  --num_workers 8 --tile_batch_size 128 \
  --use_pretrained_mae \
  --mae_checkpoint_path MAE_pretrain/MAE_epoch_500.pth.tar \
  --feature_channels 8 \
  --use_amp \
  --use_dataparallel \
  --pin_memory \
  --use_augmentation \
  --use_dropout \
  --dropout_rate 0.15
  ```
| Parameter | Description |
|-----------|-------------|
| CUDA_VISIBLE_DEVICES=0,1,2,3 | Specify GPU IDs (0-3) for multi-GPU training (adjust based on your available GPUs) |
| --train_dataset_path | Path to your custom training dataset (replace with your dataset path; demo protein datasets are provided in `train_dataset/`) |
| --output_path | Directory to save trained models, training logs, and validation results |
| --num_epochs 100 | Total training epochs (100 epochs as default for stable convergence) |
| --batch_size 8 | Batch size for training (adjust based on your GPU memory) |
| --learning_rate 0.0003 | Initial learning rate for optimizer |
| --weight_decay 5e-3 | Weight decay (L2 regularization) to prevent overfitting |
| --img_size 64 --patch_size 4 | Image size and patch size for MAE feature extraction |
| --num_workers 8 | Number of CPU workers for data loading (speeds up data preprocessing) |
| --tile_batch_size 128 | Batch size for tiled processing of large micrographs |
| --use_pretrained_mae | Enable MAE pre-trained weights initialization (critical for better performance) |
| --mae_checkpoint_path | Path to MAE pre-trained checkpoint (must match the downloaded file path) |
| --feature_channels 8 | Use 8 extra feature channels (consistent with the prediction setting) |
| --use_amp | Enable mixed precision training (reduces GPU memory usage and speeds up training) |
| --use_dataparallel | Enable multi-GPU data parallel training (works with `CUDA_VISIBLE_DEVICES`) |
| --pin_memory | Use pinned memory for data loading (accelerates GPU data transfer) |
| --use_augmentation | Enable data augmentation (random flip/rotation) to improve generalization |
| --use_dropout | Enable dropout layer in the model |
| --dropout_rate 0.15 | Dropout rate (0.15 as default to balance regularization and performance) |

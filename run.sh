conda env create -f environment.yml
# 创建环境

tmux new -s train

conda activate cryovista



# 结合mae和unet的训练，先进行MAE预训练，然后进行UNet训练

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

  
  # --enable_normalization \
  # --norm_method minmax \
  #--norm_range 0,1 \


# --use_amp 代表使用混合精度训练



# 挑选颗粒 - 默认模式（输出图像和坐标）
python /raid0_ssd/SYH/paper_code/CryoVISTA/predict_mae_unet_sam.py \
  --model_path /raid0_ssd/SYH/paper_code/CryoVISTA/output/mae_unet_integrated/models/8_extra_channel.pth \
  --mae_checkpoint_path /raid0_ssd/SYH/paper_code/CryoVISTA/MAE_pretrain/MAE_epoch_500.pth.tar \
  --input_path test_dataset \
  --output_path output/prediction_results \
  --img_size 64 --patch_size 4 \
  --feature_channels 8 \
  --sam_model_type vit_h \
  --pred_tile_bs 64 \
  --device cuda \
  --no_images



# 挑选颗粒 - 仅输出坐标（不生成图像，更快）



# --pred_tile_bs 代表一次处理多少块瓦片
# --output_path 指定输出目录
# --no_images 仅输出坐标，不生成可视化图像（提升处理速度）
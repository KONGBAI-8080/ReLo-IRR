#!/bin/bash
set -e

# 【your wandb api key】
export WANDB_API_KEY=xxx

# 日志目录
LOG_DIR="/path/log" # 【your log dir】
mkdir -p $LOG_DIR

# 训练参数
PRETRAINED_MODEL="/path/model/FLUX.1-Kontext-dev" #【your model location】
OUTPUT_DIR="SIRRCheckPoints"
REFDET_PRETRAINED_MODEL="/path/model/RDNetRRNetModels/weights/RD.pth" #【your model location】
REFDET_BACKBONE_PRETRAINED_MODEL="/path/model/EfficientNet/efficientnet-b3-5fb5a3c3.pth" #【your model location】
DATASET_NAME="/path/datasets/reflection-removal" #【your dataset location】
IMAGE_COLUMN="transmission_layer"
COND_IMAGE_COLUMN="blended"
CAPTION_COLUMN="caption"
ASPECT_RATIO_BUCKETS="672,1568;688,1504;720,1456;752,1392;800,1328;832,1248;880,1184;944,1104;1024,1024;1104,944;1184,880;1248,832;1328,800;1392,752;1456,720;1504,688;1568,672"
TRAIN_BATCH_SIZE=7
GUIDANCE_SCALE=1
GRAD_ACC_STEPS=1
OPTIMIZER="adamw"
LEARNING_RATE=5e-5
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=200
MAX_TRAIN_STEPS=7800
RANK=16
LORA_ALPHA=4
SEED=0

# Owen718/Kontext-Lora-Trainer 学习率调度策略

# TRAIN_BATCH_SIZE=8
# GUIDANCE_SCALE=1
# GRAD_ACC_STEPS=1
# OPTIMIZER="adamw"
# LEARNING_RATE=5e-5
# LR_SCHEDULER="constant"
# LR_WARMUP_STEPS=200
# MAX_TRAIN_STEPS=2600
# RANK=256
# LORA_ALPHA=256
# SEED=42

# 启动训练\  --instance_data_dir "./meta_data" 存放原数据
# --resume_from_checkpoint "latest" \
accelerate  launch train_dreambooth_lora_flux_kontext_sirr.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --pretrained_ref_det_path="$REFDET_PRETRAINED_MODEL" \
  --pretrained_ref_det_backbone_path="$REFDET_BACKBONE_PRETRAINED_MODEL" \
  --output_dir="$OUTPUT_DIR" \
  --dataset_name="$DATASET_NAME" \
  --image_column="$IMAGE_COLUMN" \
  --cond_image_column="$COND_IMAGE_COLUMN" \
  --caption_column="$CAPTION_COLUMN" \
  --mixed_precision="bf16" \
  --aspect_ratio_buckets="$ASPECT_RATIO_BUCKETS" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --guidance_scale=$GUIDANCE_SCALE \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --gradient_checkpointing \
  --optimizer="$OPTIMIZER" \
  --use_8bit_adam \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --rank=$RANK \
  --lora_alpha=$LORA_ALPHA \
  --seed="$SEED" \
  --resume_from_checkpoint "latest" \
  --grad_debug_steps=3 \
  --random_rotate \
  --random_flip \
  2>&1 | tee $LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log
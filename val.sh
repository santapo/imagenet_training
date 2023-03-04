#!/bin/sh

python main.py validate \
    --data.data_root="data/imagenet" \
    --data.image_size=224 \
    --data.num_workers=28 \
    --data.val_batch_size=16 \
    --data.use_tencrop_test="true" \
    --model.model_name="resnet50" \
    --model.num_classes=1000 \
    --model.pretrained="false" \
    --trainer.devices=4 \
    --trainer.accelerator="gpu" \
    --trainer.strategy="ddp" \
    --seed_everything=42 \
    --use_clearml="false" \
    --clearml_project_name="ImageNet Training" \
    --clearml_task_name="validate" \
    --ckpt_path="/root/imagenet_training/lightning_logs/version_9/checkpoints/epoch=53-step=270270.ckpt"
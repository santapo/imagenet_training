#!/bin/sh

python main.py fit \
    --data.data_root="data/imagenet" \
    --data.image_size=224 \
    --data.num_workers=16 \
    --data.train_batch_size=64 \
    --data.val_batch_size=16 \
    --data.use_tencrop_test="true" \
    --model.model_name="resnet50" \
    --model.num_classes=1000 \
    --model.pretrained="false" \
    --optimizer="SGD" \
    --optimizer.lr=0.1 \
    --optimizer.momentum=0.9 \
    --optimizer.weight_decay=0.0001 \
    --lr_scheduler="ReduceLROnPlateau" \
    --lr_scheduler.monitor="loss_epoch/val" \
    --lr_scheduler.patience="5" \
    --lr_scheduler.verbose="true" \
    --trainer.devices=4 \
    --trainer.accelerator="gpu" \
    --trainer.strategy="ddp" \
    --trainer.sync_batchnorm="true" \
    --trainer.max_epochs=120 \
    --seed_everything=42 \
    --use_clearml="true" \
    --clearml_project_name="ImageNet Training" \
    --clearml_task_name="Reproduce ResNet50"
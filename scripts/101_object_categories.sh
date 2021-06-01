#!/bin/bash
python3 ./train.py \
    --data data/101_ObjectCategories \
    --checkpoints checkpoints/1 \
    --model resnet18 \
    --classes 102 \
    --batch 64 \
    --workers 16 \
    --gpus 0,1,2,3

python3 ./train.py \
    --data data/101_ObjectCategories \
    --checkpoints checkpoints/2 \
    --checkpoint epoch=24_loss=2.04.pkl \
    --model resnet18 \
    --classes 102 \
    --batch 64 \
    --workers 16 \
    --gpus 0,1,2,3

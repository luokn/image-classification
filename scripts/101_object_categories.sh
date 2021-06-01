#!/bin/bash
python3 ./train.py \
    --data data/101_ObjectCategories \
    --checkpoints checkpoints/1 \
    --model resnet18 \
    --classes 102 \
    --batch 64 \
    --workers 12 \
    --gpu 1

python3 ./train.py \
    --data data/101_ObjectCategories \
    --checkpoints checkpoints/0 \
    --checkpoint epoch=83_loss=1.39.pkl \
    --model resnet18 \
    --classes 102 \
    --batch 64 \
    --workers 12 \
    --epochs 200 \
    --gpus 0,1

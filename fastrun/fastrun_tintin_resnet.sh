#! bin/sh

python train.py --dataset imagenet --device tintin --model resnet --cuda 1 --batch 128 --resume True --weighttime 2024_06_19_16_21_35 --weightname resnet_3_0.5185280584468043

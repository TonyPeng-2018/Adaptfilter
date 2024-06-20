#! bin/sh

python train.py --dataset imagenet --device tintin --model mobilenetV2 --cuda 0 --batch 128 --resume True --weighttime 2024_06_19_16_20_07 --weightname mobilenetV2_9_0.550155522512713

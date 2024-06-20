#! bin/sh

python train.py --dataset imagenet --device tintin --model mobilenetV3 --mobilev3size large --cuda 2 --batch 128

#! bin/sh

# python train.py --dataset imagenet --device tintin --model mobilenetV3 --cuda 3 --batch 128 --modeltime 2024_06_19_16_35_55 --modelname mobilenetV3_10_0.5137433507783931
python train.py --dataset imagenet --device tintin --model mobilenetV3 --cuda 2 --batch 128 --resume True --weighttime 2024_06_19_16_34_43 --weightname mobilenetV3_13_0.48028161868299557

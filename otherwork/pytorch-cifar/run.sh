#! bin/sh
# run all the experiments

python main.py --modelname ResNet18 > ResNet18.txt 
python main.py --modelname ResNet34 > ResNet34.txt 
python main.py --modelname ResNet50 > ResNet50.txt 
python main.py --modelname ResNet101 > ResNet101.txt --device cuda:0
python main.py --modelname ResNet152 > ResNet152.txt --device cuda:1
python main.py --modelname ShuffleNetV2 > ShuffleNetV2.txt --device cuda:2 
python main.py --modelname SimpleDLA > SimpleDLA.txt 
python main.py --modelname MobileNetV2 > MobileNetV2.txt 

python main_tintin.py --modelname ResNet18 > ResNet18.txt --device cuda:0
python main_tintin.py --modelname ResNet34 > ResNet34.txt --device cuda:1
python main_tintin.py --modelname ResNet50 > ResNet50.txt --device cuda:2
python main_tintin.py --modelname ResNet101 > ResNet101.txt --device cuda:3
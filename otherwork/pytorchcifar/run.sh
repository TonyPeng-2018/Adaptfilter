#! bin/sh
# run all the experiments

python main.py --modelname ResNet18 > ResNet18.txt 
python main.py --modelname ResNet34 > ResNet34.txt 
python main.py --modelname ResNet50 > ResNet50.txt 
python main.py --modelname ResNet101 > ResNet101.txt 
python main.py --modelname ResNet152 > ResNet152.txt 
python main.py --modelname ShuffleNetV2 > ShuffleNetV2.txt 
python main.py --modelname SimpleDLA > SimpleDLA.txt 
python main.py --modelname MobileNetV2 > MobileNetV2.txt
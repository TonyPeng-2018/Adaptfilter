#! bin/sh
# run all the experiments

python main.py --modelname ResNet18 > ResNet18.log 
python main.py --modelname ResNet34 > ResNet34.log 
python main.py --modelname ResNet50 > ResNet50.log 
python main.py --modelname ResNet101 > ResNet101.log 
python main.py --modelname ResNet152 > ResNet152.log 
python main.py --modelname ShuffleNetV2 > ShuffleNetV2.log 
python main.py --modelname SimpleDLA > SimpleDLA.log 
python main.py --modelname MobileNetV2 > MobileNetV2.log
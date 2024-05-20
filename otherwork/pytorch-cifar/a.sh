#! bin/sh
# run all the experiments

python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname ResNet18 > ResNet18.txt 
python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname ResNet34 > ResNet34.txt 
python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname ResNet50 > ResNet50.txt 
python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname ResNet101 > ResNet101.txt 
python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname ResNet152 > ResNet152.txt 
python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname ShuffleNetV2 > ShuffleNetV2.txt 
python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname SimpleDLA > SimpleDLA.txt 
python ~/Workspace/Adaptfilter/otherwork/pytorch-cifar/main.py --modelname MobileNetV2 > MobileNetV2.txt 
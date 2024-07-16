import random
# random seed
random.seed(2024)
# image
imagenet_path = '/data/anp407/imagenet/ILSVRC/Data/CLS-LOC/train/'
imagenet20_path = '/data/anp407/imagenet-20/'

import os 
if not os.path.exists(imagenet20_path):
    os.makedirs(imagenet20_path)

# get all folders at imagenet path
folders = os.listdir(imagenet_path)
# shuffle
random.shuffle(folders)
# select 20 folders
folders = folders[:20]
# create the 20 folders
for folder in folders:
    os.system('cp -r ' + imagenet_path + folder + ' ' + imagenet20_path)

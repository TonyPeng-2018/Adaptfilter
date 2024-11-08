dataset = 'data/imagenet-20-new/train/'
import os 
folder_list = os.listdir(dataset)
print(folder_list)

import csv
csv_file = '/home/tonypeng/Workspace1/adaptfilter/data/imagenet/LOC_val_solution.csv'
testset = 'data/imagenet-20-new/test/'
rawset = '/home/tonypeng/Workspace1/adaptfilter/data/imagenet/ILSVRC/Data/CLS-LOC/val/'
with open(csv_file, newline='') as file:
    spamreader = csv.reader(file, delimiter=',')
    for row in spamreader:
        img = row[0] + '.JPEG'
        label = row[1][:9]
        if label in folder_list:
            if not os.path.exists(testset + label):
                os.makedirs(testset + label)
            os.system('cp ' + rawset + img + ' ' + testset + label + '/' + img)
            
    
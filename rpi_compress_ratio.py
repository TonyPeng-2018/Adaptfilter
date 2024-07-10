# Here we need to plot a latency vs accuracy


from Dataloaders import dataloader_cifar10, dataloader_imagenet
from datetime import datetime
from DeepN import main_no_store
import numpy as np
from PIL import Image
import seaborn as sns
import subprocess 
import time
from tqdm import tqdm

from Utils import utils

# load 100 image
imgpath_root = '../data/cifar-10-client/'
imgpath = [imgpath_root +  str(i) + '.bmp' for i in range(100)]
out_imgpath = [imgpath_root + str(i) for i in range(100)]
compress_ratio = [x for x in range(5, 100, 5)]
stime = datetime.now().strftime('%m-%d %H:%M')
logger = utils.APLogger(path='./Logs/cifar-10/baseline_plot_accuracy_vs_latency_' + stime + '.log')
logger.write('plot accuracy vs latency for cifar-10 dataset\n')

latency = []
accuracy = []

latency_mat = np.zeros((100, 19))

for ind, ip in tqdm(enumerate(imgpath)):
    for ind2, cr in enumerate(compress_ratio):
        img = Image.open(ip)
        img = np.array(img)
        time1 = time.time()
        qtable = main_no_store(img)
        time2 = time.time()
        # store qtable
        np.savetxt('qt.txt', qtable, fmt='%d')
        command = 'cjpeg -dct int -qtable qt.txt -baseline'
        command += ' -quality ' + str(cr)
        command += ' -outfile '
        command += str(out_imgpath[ind])+'_'+str(cr)+'.jpg'
        command += ' ' + str(ip)
        
        command = command.split(' ')
        time3 = time.time()
        subprocess.run(command)
        time4 = time.time()

        # network latency
        
        latency_mat[ind, ind2] = time4-time3 + time2-time1
np.savez('latency_mat.npz', latency_mat=latency_mat)
        
        
        
    


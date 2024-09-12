from Dataloaders import dataloader_image_20
_, test, _, _ = dataloader_image_20.Dataloader_imagenet_20_integrated(test_batch=1, transform=False)

import cv2
root_path = '../data/imagenet-20-client/'
label_f = open(root_path+'label.txt', 'w') 
import numpy as np
for i, (data, label, _) in enumerate(test):
    # store first 600 images to bmp and store label
    data = data.squeeze(0)
    data = data.permute(1, 2, 0)
    # not correct color
    data = data.numpy()
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    
    # store in float
    data = data*255
    data = data.astype(np.uint8)
    # store in bmp
    cv2.imwrite(root_path+'%d.bmp'%i, data)
    label_f.write('%d\n'%(label))
    if i== 599:
        break
label_f.close()



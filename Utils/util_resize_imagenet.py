import cv2
import os 

# image
in_path = '../../data/imagenet-20-raw-image/'
out_path = '../../data/imagenet-20-raw-image-224/'

if not os.path.exists(out_path):
    os.makedirs(out_path)

# get all folders at imagenet path
count = 600
# cv2 read image and resize
for c in range(count):
    img= cv2.imread(in_path+'/'+str(c)+'.bmp')
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(out_path+'/'+str(c)+'.bmp', img)

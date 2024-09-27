# read a 32*32*3 image and use png to encode it
import numpy as np
import cv2
import sys
import base64
import time

image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
t1 = time.time()
# encode the image
_, png = cv2.imencode('.png', image)
png = png.tobytes()
png = base64.b64encode(png)
t2 = time.time()
tt = (t2-t1)*1000
print('png encoding time:', tt)

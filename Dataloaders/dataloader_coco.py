# we can simply use the coco api to load the data

import torchvision
from PIL import Image
# import transforms
import torchvision.transforms as transforms

dataDir = './cocoapi/'
dataType = 'val2017' # 以验证集为例
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
print(annFile)

# torchvision.datasets.CocoDetection(root: str, annFile: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None)
def Dataloader_coco(r_path, a_path, ):
    cocodataset = torchvision.datasets.CocoDetection(
        root=r_path, 
        annFile=a_path,
        transform=transforms.ToTensor())
    return cocodataset
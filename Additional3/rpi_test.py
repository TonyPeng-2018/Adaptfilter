# load the client model from mobilenetV2
from Models import mobilenetv2
import time
from tqdm import tqdm
# load the client model from mobilenetV2
client, _ = mobilenetv2.stupid_model_splitter(num_classes=10,
                                              weight_path='./Weights/MobileNetV2.pth',
                                              device = 'cpu')

# load a image from data
from Dataloaders import dataloader_cifar10
train, test, _ = dataloader_cifar10.Dataloader_cifar10(train_batch=1,
                                                    datasetpath = '/home/pi302/Workspace/data/',
                                                       num_workers = 2)
import torch
client_8 = torch.ao.quantization.quantize_dynamic(
    client,  # the original model
    {torch.nn.Conv2d},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
client_8.eval()
print(client)


with torch.no_grad():
    count = 0
    s_time = time.time()
    for i, (data, target) in tqdm(enumerate(train)):
        count += data.size(0)
        output = client(data)
        if count >= 1000:
            break
    print((time.time() - s_time)/count)
    


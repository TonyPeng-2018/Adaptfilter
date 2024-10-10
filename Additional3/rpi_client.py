# this file is for device on the client side

# load the dataset

# This file is for trainning
# Run this on the server, or as we called offline.

import argparse
import base64
import cv2
import datetime
from Models import mobilenetv2, mobilenetv3, resnet, gatedmodel
import numpy as np
import os
import PIL
import sys
import time
import torch
from tqdm import tqdm
from Utils import utils, encoder


def main(args):
    p_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logger = utils.APLogger(
        "./Logs/" + args.dataset + "/client_" + p_start_time + ".log\n"
    )

    # write the logger with all args parameters
    logger.write("model: %s, dataset: %s\n" % (args.model, args.dataset))
    logger.write(
        "batch: %d, compressor: %s, ranker: %s\n"
        % (args.batch, args.compressor, args.ranker)
    )
    logger.write("weight: %s\n" % (args.weight))

    # 1. load the dataset
    weight_path = "./Weights/" + args.dataset + "/client/" + args.model + ".pth"
    weight_root = "./Weights/" + args.dataset + "/"

    if args.dataset == "cifar-10":
        num_classes = 10
    elif args.dataset == "cifar-100":
        num_classes = 100
    elif args.dataset == "imagenet-mini":
        num_classes = 100
    elif args.dataset == "imagenet-tiny":
        num_classes = 200
    elif args.dataset == "imagenet":
        num_classes = 1000

    if args.model == "mobilenetV2":
        c_model = mobilenetv2.mobilenetv2_splitter_client(
            num_classes=num_classes, weight_root=weight_root, device="cpu"
        )

    elif args.model == "mobilenetV3":
        c_model = mobilenetv3.mobilenetv3_splitter_client(
            num_classes=num_classes, weight_root=weight_root, device="cpu"
        )

    elif args.model == "resnet":
        c_model = resnet.resnet_splitter_client(
            num_classes=num_classes,
            weight_root=weight_root,
            device="cpu",
            layers=args.resnetsize,
        )

    c_model.eval()
    c_model = torch.ao.quantization.quantize_dynamic(
        c_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    # 2. dataset
    # directly read bmp image from the storage
    data_root = "../data/" + args.dataset + "-client/"
    images_list = os.listdir(data_root)
    # remove the label file
    images_list.remove("labels.txt")
    images_list = sorted(images_list)

    JPEG_time_25 = 0
    JPEG_time_50 = 0
    JPEG_time_75 = 0
    CJPEG_time = 0

    image_batch = 1
    image_bucket = []
    image_count = 0

    i_stop = 60

    for i, i_path in tqdm(enumerate(images_list)):
        if i >= i_stop:
            break

        image_path = data_root + i_path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0

        image_count += 1
        image_bucket.append(image)
        if image_count < image_batch:
            continue
        else:
            image_count = 0

        # 3. compress the image
        # 3.1 JPEG 25
        s_time = time.time()
        for image in image_bucket:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
            # result, encimg = cv2.imencode('.jpg', image, encode_param)
            # store the image
            cv2.imwrite("./jpeg_25.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
            # encoder.simple_run('./jpeg_25.jpg', './jpeg_25.out')
            # with open('./jpeg_25.out', 'rb') as f:
            #     c_encode = f.readlines()[0]

        e_time = time.time()
        JPEG_time_25 += e_time - s_time

        # 3.2 JPEG 50
        s_time = time.time()
        for image in image_bucket:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            # result, encimg = cv2.imencode('.jpg', image, encode_param)
            e_time = time.time()
            # cv2.imwrite('./jpeg_50.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            # encoder.simple_run('./jpeg_50.jpg', './jpeg_50.out')
            # with open('./jpeg_50.out', 'rb') as f:
            #     c_encode = f.readlines()[0]

        JPEG_time_50 += e_time - s_time

        # 3.3 JPEG 75
        s_time = time.time()
        for image in image_bucket:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            # result, encimg = cv2.imencode('.jpg', image, encode_param)
            cv2.imwrite("./jpeg_75.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            # encoder.simple_run('./jpeg_75.jpg', './jpeg_75.out')
            # with open('./jpeg_75.out', 'rb') as f:
            #     c_encode = f.readlines()[0]

        e_time = time.time()
        JPEG_time_75 += e_time - s_time

        # 3.4 progressive JPEG
        s_time = time.time()
        for image in image_bucket:
            encode_param = [int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
            # result, encimg = cv2.imencode('.jpg', image, encode_param)
            cv2.imwrite("./c_jpeg.jpg", image, [int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
            # encoder.simple_run('./c_jpeg.jpg', './c_jpeg.out')
            # with open('./c_jpeg.out', 'rb') as f:
            #     c_encode = f.readlines()[0]

        e_time = time.time()
        CJPEG_time += e_time - s_time

        image_bucket = []

    JPEG_time_25 /= i_stop
    JPEG_time_50 /= i_stop
    JPEG_time_75 /= i_stop
    CJPEG_time /= i_stop

    print(
        "JPEG 25: %.3f, JPEG 50: %.3f, JPEG 75: %.3f, CJPEG: %.3f\n"
        % (JPEG_time_25, JPEG_time_50, JPEG_time_75, CJPEG_time)
    )


if __name__ == "__main__":
    print("enter")
    parser = argparse.ArgumentParser()
    # we need the name of model, the name of dataset
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument(
        "--compressor", type=str, default="home", help="compressor name"
    )
    parser.add_argument("--dataset", type=str, default="imagenet", help="dataset name")
    parser.add_argument(
        "--mobilev3size", type=str, default="small", help="the size of the mobilev3"
    )
    parser.add_argument("--model", type=str, default="mobilenetV2", help="model name")
    parser.add_argument("--ranker", type=str, default="zeros", help="ranker name")
    parser.add_argument("--resnetsize", type=int, default=18, help="resnet layers")
    parser.add_argument(
        "--weight", type=str, default="./Weight/cifar-10/", help="weight path"
    )
    args = parser.parse_args()
    print(args)
    main(args)

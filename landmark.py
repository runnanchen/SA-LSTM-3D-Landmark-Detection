from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms, utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import os
import math
from copy import deepcopy
import pandas as pd
from MyDataLoader import Rescale, ToTensor, LandmarksDataset
import MyModel
import TrainNet
import LossFunction

plt.ion()  # interactive mode
batchSize = 1
landmarkNum = 17
image_scale = (72, 96, 96)
original_image_scale = (576, 768, 768)
cropSize = (32, 32, 32)
use_gpu = 0
iteration = 3
traincsv = 'skull_train1_m_mini.csv'
testcsv = 'skull_test1_m_mini.csv'

dataRoot = "processed_data/"
epochs = 1000
saveName = "test"
testName = "190VGG19_bn_concatFPN_originOff_withIceptionkernel_newdata_32_noPretrain_try.pkl"

fine_LSTM = MyModel.fine_LSTM(landmarkNum, use_gpu, iteration, cropSize).cuda(use_gpu)
corseNet = MyModel.coarseNet(landmarkNum, use_gpu, image_scale).cuda(use_gpu)

print("image scale ", image_scale)

print("GPU: ", use_gpu)
print(saveName)

transform_origin = transforms.Compose([
    Rescale(image_scale),
    ToTensor()
])

train_dataset_origin = LandmarksDataset(csv_file=dataRoot + traincsv,
                                        root_dir=dataRoot + "images",
                                        transform=transform_origin,
                                        landmarksNum=landmarkNum
                                        )

val_dataset = LandmarksDataset(csv_file=dataRoot + testcsv,
                               root_dir=dataRoot + "images",
                               transform=transform_origin,
                               landmarksNum=landmarkNum
                               )

dataloaders = {}
train_dataloader = []
val_dataloader = []

train_dataloader_t = DataLoader(train_dataset_origin, batch_size=batchSize,
                                shuffle=False, num_workers=0)

for data in train_dataloader_t:
    train_dataloader.append(data)

val_dataloader_t = DataLoader(val_dataset, batch_size=batchSize,
                              shuffle=False, num_workers=4)

for data in val_dataloader_t:
    val_dataloader.append(data)

print(len(train_dataloader), len(val_dataloader))

train_dataloader_t = ''
val_dataloader_t = ''

dataloaders = {'train': train_dataloader_t, 'val': val_dataloader}

criterion_coarse = LossFunction.coarse_heatmap(use_gpu, batchSize, landmarkNum, image_scale)

params = list(corseNet.parameters()) + list(fine_LSTM.parameters())

optimizer_ft = optim.Adam(params)

TrainNet.train_model(corseNet, fine_LSTM, dataloaders, criterion_coarse,
                     optimizer_ft, epochs, use_gpu, saveName, landmarkNum, image_scale)

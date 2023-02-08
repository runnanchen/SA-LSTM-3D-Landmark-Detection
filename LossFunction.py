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
from skimage import io, transform
import math
from copy import deepcopy
import pandas as pd
import math
import copy
import time
import PIL
# import angle
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
import MyUtils
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):

    #print (recon_x.view(-1))
    #BCE = F.binary_cross_entropy(recon_x.view(1,-1), x.view(1,-1), size_average = False)
    # BCE = F.mse_loss(recon_x.view(1,-1), x.view(1,-1), size_average = False)
    BCE = F.l1_loss(recon_x.view(1,-1), x.view(1,-1), size_average = False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # return BCE + KLD
    return BCE


class attentionLoss(nn.Module):
    def __init__(self, gpu):
        super(attentionLoss, self).__init__()
        self.lossFun = torch.nn.L1Loss(size_average=False)
        self.gpu = gpu
        self.delta = torch.tensor([0.]).cuda(self.gpu)

    def forward(self, coormeanAngles, labelsAngles, attention):
        topN = coormeanAngles.size()[0]
        topkP, indexs = torch.topk(attention, topN)
        indexs = indexs.cpu().numpy()
        loss = 0
        alll = 0
        for i in range(topN):
            temp = attention[indexs[i]]
            alll += temp
            # ~ temp = 1
            loss += temp * self.lossFun(coormeanAngles[i, :], labelsAngles[i, :])
        # ~ loss += self.lossFun(1/(torch.var(attention)*1e8 + 1), self.delta)
        # ~ print (alll/topN - 1/ 5456)
        # ~ print (torch.var(attention))
        return loss

class coarse_heatmap(nn.Module):
    def __init__(self, config):
        # use_gpu, batchSize, landmarkNum, image_scale
        super(coarse_heatmap, self).__init__()
        self.use_gpu = config.use_gpu
        self.batchSize = config.batchSize
        self.landmarkNum = config.landmarkNum
        self.l1Loss = nn.L1Loss(size_average=False)
        self.Long, self.higth, self.width = config.image_scale
        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        self.HeatMap_groundTruth = torch.zeros(self.Long * 2, self.higth * 2, self.width * 2).cuda(self.use_gpu)


        rr = 21
        dev = 2
        referPoint = (self.Long, self.higth, self.width)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap_groundTruth[k][i][j] = math.exp(-1 * temdis**2 / (2 * dev**2))

    def forward(self, predicted_heatmap, local_coordinate, labels, phase):
        loss = 0
        labels_b = labels * torch.tensor([self.higth - 1, self.width - 1, self.Long - 1]).cuda(self.use_gpu)
        labels_b = np.round(labels_b.detach().cpu().numpy()).astype("int")
        X, Y, Z = labels_b[0, :, 0], labels_b[0, :, 1], labels_b[0, :, 2]
        index = [2, 0, 1]

        for i in range(self.landmarkNum):

            coarse_heatmap = self.HeatMap_groundTruth[self.Long - Z[i]: 2 * self.Long - Z[i],
                                                                        self.higth - X[i]: 2 * self.higth - X[i],
                                                                        self.width - Y[i]: 2 * self.width - Y[i]]

            loss += torch.abs(predicted_heatmap[i] - coarse_heatmap / (coarse_heatmap.sum())).sum()
        return loss


class coarse_heatmap_b(nn.Module):
    def __init__(self, use_gpu, batchSize, landmarkNum, image_scale):
        super(coarse_heatmap_b, self).__init__()
        self.use_gpu = use_gpu
        self.batchSize = batchSize
        self.landmarkNum = landmarkNum

    def forward(self, predicted_heatmap, local_coordinate, labels, ROIs_b, size_tensor, phase):
        loss = 0
        index = [2, 0, 1]
        labels_b = labels[0, :, index]

        for i in range(self.landmarkNum):
            # loss += ((torch.abs(local_coordinate - labels_b[i, :]) * size_tensor[0][index]).permute(3, 0, 1, 2) * predicted_heatmap[i]).sum()
            heatmap = F.sigmoid(predicted_heatmap[i])
            heatmap = heatmap / heatmap.sum()
            weigthmap = torch.abs(local_coordinate - labels_b[i, :])
            # mean = torch.sum((local_coordinate.permute(3, 0, 1, 2) * heatmap).view(3, -1), dim = 1)
            # delta = (torch.pow(local_coordinate - mean, 2).permute(3, 0, 1, 2) * heatmap).sum()
            # weigthmap[weigthmap < 0.003] = weigthmap[weigthmap < 0.003] * 0
            # print(torch.pow(1 - delta, 2))
            # print(torch.abs((weigthmap.permute(3, 0, 1, 2) * heatmap).sum()))
            # loss += torch.abs((weigthmap.permute(3, 0, 1, 2) * heatmap).sum()) + torch.pow(1 - delta, 2)
            loss += torch.abs((weigthmap.permute(3, 0, 1, 2) * heatmap).sum())
            # loss += torch.abs((torch.pow(local_coordinate - labels_b[i, :], 2).permute(3, 0, 1, 2) * heatmap).sum())
        return loss

class fine_heatmap_b(nn.Module):
    def __init__(self, use_gpu, batchSize, landmarkNum, cropSize):
        super(fine_heatmap_b, self).__init__()
        self.use_gpu = use_gpu
        self.batchSize = batchSize
        self.landmarkNum = landmarkNum

    def forward(self, predicted_heatmap, coordinate, labels, ROIs_b, size_tensor, phase):
        loss = 0
        for i in range(self.landmarkNum):
            heatmap = F.sigmoid(predicted_heatmap[i]) + 1e-10
            heatmap = heatmap / (heatmap.sum())
            loss += (torch.abs(coordinate[i] - labels[0, i, :]).permute(3, 0, 1, 2) * heatmap).sum()

        return loss

class fine_heatmap(nn.Module):
    def __init__(self, config):
        # use_gpu, batchSize, landmarkNum, cropSize
        super(fine_heatmap, self).__init__()
        self.use_gpu = config.use_gpu
        self.batchSize = config.batchSize
        self.landmarkNum = config.landmarkNum
        self.l1Loss = nn.L1Loss(size_average=False)

        self.Long, self.higth, self.width = config.crop_size
        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        self.HeatMap_groundTruth = torch.zeros(self.Long, self.higth, self.width).cuda(self.use_gpu)

        rr = 11
        dev = 2
        referPoint = (self.Long//2, self.higth//2, self.width//2)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap_groundTruth[k][i][j] = math.exp(-1 * temdis**2 / (2 * dev**2))

    def forward(self, predicted_heatmap):
        loss = 0
        for i in range(self.landmarkNum):

            loss += self.binaryLoss(predicted_heatmap[i], self.HeatMap_groundTruth).sum()
        return loss

class fine_offset(nn.Module):
    def __init__(self, use_gpu, batchSize, landmarkNum):
        super(fine_offset, self).__init__()
        self.use_gpu = use_gpu
        self.batchSize = batchSize
        self.landmarkNum = landmarkNum
        self.l1Loss = nn.L1Loss(size_average=False)

    def forward(self, predicted_coordinate, local_coordinate, labels, ROIs_b, base_coordinate, phase):
        loss = 0
        # print('predict ', predicted_point_offsets[0])
        # print(labels[2,:])
        # print('gt ', torch.mean(predicted_point_offsets[2]+ point_coordinate[2], dim=0))

        # for i in range(self.landmarkNum):
        #     repeat_labels = labels[i, :].unsqueeze(0).repeat(predicted_point_offsets[i].size()[0], 1)
        #     loss += self.l1Loss(predicted_point_offsets[i], repeat_labels)
            # print(point_coordinate[i])
        # return loss
        for i in range(self.landmarkNum):
            # repeat_labels = labels[0, i, :].repeat(32, 32, 32, 1)
            # print(repeat_labels.size())
            # print(labels[0, i, :])
            # print(repeat_labels[3, 3, 1, :])
            # print(repeat_labels[15, 10, 2, :])


            # repeat_ROIs = ROIs[i, :].unsqueeze(0).repeat(predicted_point_offsets[i].size()[0], 1)
            # # print("repeat_labels", repeat_labels.size())
            # # print(labels[i, :].unsqueeze(0).detach().cpu().numpy() * np.array([767, 767, 575]))
            if phase == 'val':
            #     print(labels)
            #     print((torch.abs(local_coordinate[i] - labels[0, i, :]) - torch.abs(predicted_coordinate[i] - labels[0, i, :])).view(-1, 3)[0:30, :].detach().cpu().numpy() * np.array([767, 767, 575]))
            #     print((local_coordinate[i] - labels[0, i, :]).view(-1, 3)[0:30, :].detach().cpu().numpy() * np.array([767, 767, 575]))
                print()
                print((labels[0, i, :] - local_coordinate[i]).view(-1, 3)[320:352, :].detach().cpu().numpy() * np.array([767, 767, 575]))
                print((predicted_coordinate[i] - local_coordinate[i]).view(-1, 3)[320:352, :].detach().cpu().numpy() * np.array([767, 767, 575]))
                print()
                # print((local_coordinate[i])1.view(-1, 3)[320:352, :].detach().cpu().numpy() * np.array([767, 767, 575]))
            # if phase == 'val':
            #     print((torch.abs(predicted_point_offsets[i] - repeat_labels))[0:30, :].detach().cpu().numpy() * np.array([767, 767, 575]))

            loss += torch.abs(predicted_coordinate[i] - labels[0, i, :]).sum()
            # # print(point_coordinate[i])
        return loss

class fusionLossFunc_improved(nn.Module):
    def __init__(self, use_gpu, R, imageSize, imageNum, landmarkNum):
        super(fusionLossFunc_improved, self).__init__()

        l, h, w = 72, 96, 96
        # ~ l, h, w = 144, 192, 192

        self.use_gpu = use_gpu
        self.R = R
        self.width = w
        self.higth = h
        self.Long = l

        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        # ~ self.binaryLoss = nn.BCEWithLogitsLoss()

        self.huberLoss = torch.nn.L1Loss()
        # ~ self.offsetMask = torch.zeros(h, w).cuda(self.use_gpu)

        self.offsetMapx = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapy = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapz = np.ones((self.Long * 2, self.higth * 2, self.width * 2))

        self.HeatMap = np.zeros((self.Long * 2, self.higth * 2, self.width * 2))

        self.binary_class_groundTruth = Variable(torch.zeros(imageNum, landmarkNum, l, h, w).cuda(self.use_gpu))

        rr = R
        referPoint = (self.Long, self.higth, self.width)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap[k][i][j] = 1

        for i in range(2 * self.Long):
            self.offsetMapz[i, :, :] = self.offsetMapz[i, :, :] * i

        for i in range(2 * self.higth):
            self.offsetMapx[:, i, :] = self.offsetMapx[:, i, :] * i

        for i in range(2 * self.width):
            self.offsetMapy[:, :, i] = self.offsetMapy[:, :, i] * i

        self.offsetMapz = referPoint[0] - self.offsetMapz
        self.offsetMapx = referPoint[1] - self.offsetMapx
        self.offsetMapy = referPoint[2] - self.offsetMapy

        # print (self.HeatMap)
        # print (self.offsetMapx)
        # print (self.offsetMapy)

        self.HeatMap = Variable(torch.from_numpy(self.HeatMap)).cuda(self.use_gpu).float()

        self.offsetMapx = Variable(torch.from_numpy(self.offsetMapx)).cuda(self.use_gpu).float() / rr
        self.offsetMapy = Variable(torch.from_numpy(self.offsetMapy)).cuda(self.use_gpu).float() / rr
        self.offsetMapz = Variable(torch.from_numpy(self.offsetMapz)).cuda(self.use_gpu).float() / rr

        return

    def forward(self, featureMaps, landmarks):
        # ~ print (featureMaps.size())

        imageNum = featureMaps[0].size()[0]
        # ~ landmarkNum = int(featureMaps[0].size()[1]/3)
        landmarkNum = int(featureMaps[0].size()[1])
        # ~ landmarkNum = int(featureMaps[0].size()[1]/4)

        l, h, w = featureMaps[0].size()[2], featureMaps[0].size()[3], featureMaps[0].size()[4]
        # ~ print ("size: ", featureMaps[0].size())
        # ~ print ()
        lossOff = 0
        lossReg = 0
        loss = 0
        # print("1")
        X = np.round((landmarks[:, :, 0] * (h - 1)).numpy()).astype("int")
        Y = np.round((landmarks[:, :, 1] * (w - 1)).numpy()).astype("int")
        Z = np.round((landmarks[:, :, 2] * (l - 1)).numpy()).astype("int")

        for imageId in range(imageNum):
            for landmarkId in range(landmarkNum):
                # ~ print (Z[imageId][landmarkId], X[imageId][landmarkId], Y[imageId][landmarkId], self.HeatMap.size(), landmarkNum)
                # ~ print (l - Z[imageId][landmarkId], 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId], 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId], 2*w - Y[imageId][landmarkId])

                # ~ MyUtils.showDICOM(self.HeatMap[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]].detach().cpu().numpy(), X[imageId][landmarkId], Y[imageId][landmarkId], Z[imageId][landmarkId])

                self.binary_class_groundTruth[imageId, landmarkId, :, :, :] = self.HeatMap[
                                                                              l - Z[imageId][landmarkId]: 2 * l -
                                                                                                          Z[imageId][
                                                                                                              landmarkId],
                                                                              h - X[imageId][landmarkId]: 2 * h -
                                                                                                          X[imageId][
                                                                                                              landmarkId],
                                                                              w - Y[imageId][landmarkId]: 2 * w -
                                                                                                          Y[imageId][
                                                                                                              landmarkId]]
            # self.offsetMask = temMap + self.offsetMask

        indexs = self.binary_class_groundTruth > 0
        # indexs = self.offsetMask > 0
        # indexs = getMask(self.binary_class_groundTruth)
        # print("3")
        temloss = [
            [self.binaryLoss(featureMaps[0][imageId][landmarkId], self.binary_class_groundTruth[imageId][landmarkId])]
            # ~ , \

            # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*1][indexs[imageId][landmarkId]], \
            # ~ self.offsetMapx[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]) , \

            # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*2][indexs[imageId][landmarkId]], \
            # ~ self.offsetMapy[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]), \

            # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*3][indexs[imageId][landmarkId]], \
            # ~ self.offsetMapz[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]])]

            for imageId in range(imageNum)
            for landmarkId in range(landmarkNum)]
        loss1 = (sum([sum(temloss[ind]) for ind in range(imageNum * landmarkNum)])) / (imageNum * landmarkNum)
        # print("4")

        return loss1


class fusionLossFunc_improved_2(nn.Module):
    def __init__(self, use_gpu, R, imageSize, imageNum, landmarkNum):
        super(fusionLossFunc_improved_2, self).__init__()

        # ~ l, h, w = 96, 96, 96
        # ~ l, h, w = 48, 48, 48
        l, h, w = 64, 64, 64

        self.use_gpu = use_gpu
        self.R = R
        self.width = w
        self.higth = h
        self.Long = l

        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        # ~ self.binaryLoss = nn.BCEWithLogitsLoss()

        self.huberLoss = torch.nn.L1Loss()
        # ~ self.offsetMask = torch.zeros(h, w).cuda(self.use_gpu)

        self.offsetMapx = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapy = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapz = np.ones((self.Long * 2, self.higth * 2, self.width * 2))

        self.HeatMap = np.zeros((self.Long * 2, self.higth * 2, self.width * 2))

        rr = R
        referPoint = (self.Long, self.higth, self.width)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap[k][i][j] = 1

        for i in range(2 * self.Long):
            self.offsetMapz[i, :, :] = self.offsetMapz[i, :, :] * i

        for i in range(2 * self.higth):
            self.offsetMapx[:, i, :] = self.offsetMapx[:, i, :] * i

        for i in range(2 * self.width):
            self.offsetMapy[:, :, i] = self.offsetMapy[:, :, i] * i

        self.offsetMapz = referPoint[0] - self.offsetMapz
        self.offsetMapx = referPoint[1] - self.offsetMapx
        self.offsetMapy = referPoint[2] - self.offsetMapy

        # print (self.HeatMap)
        # print (self.offsetMapx)
        # print (self.offsetMapy)

        # ~ self.HeatMap = Variable(torch.from_numpy(self.HeatMap)).cuda(self.use_gpu).float()

        self.binary_class_groundTruth = torch.from_numpy(
            self.HeatMap[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float()

        # ~ self.offsetMapx = Variable(torch.from_numpy(self.offsetMapx)).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapy = Variable(torch.from_numpy(self.offsetMapy)).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapz = Variable(torch.from_numpy(self.offsetMapz)).cuda(self.use_gpu).float() / rr

        return

    def forward(self, featureMaps, landmarks):

        imageNum, landmarkNum, l, h, w = featureMaps.size()
        # ~ print ("size: ", featureMaps[0].size())
        # ~ print ()
        lossOff = 0
        lossReg = 0
        loss = 0

        indexs = self.binary_class_groundTruth > 0
        # print("3")
        temloss = [[self.binaryLoss(featureMaps[imageId][landmarkId], self.binary_class_groundTruth)]
                   # ~ , \

                   # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*1][indexs[imageId][landmarkId]], \
                   # ~ self.offsetMapx[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]) , \

                   # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*2][indexs[imageId][landmarkId]], \
                   # ~ self.offsetMapy[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]), \

                   # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*3][indexs[imageId][landmarkId]], \
                   # ~ self.offsetMapz[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]])]

                   for imageId in range(imageNum)
                   for landmarkId in range(landmarkNum)]
        loss2 = (sum([sum(temloss[ind]) for ind in range(imageNum * landmarkNum)])) / (imageNum * landmarkNum)
        # print("4")

        return loss2


class fusionLossFunc_improved_2_b(nn.Module):
    def __init__(self, use_gpu, R, imageSize, imageNum, landmarkNum):
        super(fusionLossFunc_improved_2_b, self).__init__()

        # ~ l, h, w = 96, 96, 96
        # ~ l, h, w = 48, 48, 48
        l, h, w = 64, 64, 64

        self.use_gpu = use_gpu
        self.R = R
        self.width = w
        self.higth = h
        self.Long = l

        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        # ~ self.binaryLoss = nn.BCEWithLogitsLoss()

        self.huberLoss = torch.nn.L1Loss(size_average=False)
        # ~ self.offsetMask = torch.zeros(h, w).cuda(self.use_gpu)

        self.offsetMapx = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapy = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapz = np.ones((self.Long * 2, self.higth * 2, self.width * 2))

        self.HeatMap = np.zeros((self.Long * 2, self.higth * 2, self.width * 2))

        rr = R
        referPoint = (self.Long, self.higth, self.width)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap[k][i][j] = 1

        for i in range(2 * self.Long):
            self.offsetMapz[i, :, :] = self.offsetMapz[i, :, :] * i

        for i in range(2 * self.higth):
            self.offsetMapx[:, i, :] = self.offsetMapx[:, i, :] * i

        for i in range(2 * self.width):
            self.offsetMapy[:, :, i] = self.offsetMapy[:, :, i] * i

        self.offsetMapz = referPoint[0] - self.offsetMapz
        self.offsetMapx = referPoint[1] - self.offsetMapx
        self.offsetMapy = referPoint[2] - self.offsetMapy

        # print (self.HeatMap)
        # print (self.offsetMapx)
        # print (self.offsetMapy)

        # ~ self.HeatMap = Variable(torch.from_numpy(self.HeatMap)).cuda(self.use_gpu).float()

        self.binary_class_groundTruth = torch.from_numpy(
            self.HeatMap[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float()

        # ~ self.offsetMapx = torch.from_numpy(self.offsetMapx[l - l//2: 2*l - l//2, h - h//2: 2*h - h//2, w - w//2: 2*w - w//2]).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapy = torch.from_numpy(self.offsetMapy[l - l//2: 2*l - l//2, h - h//2: 2*h - h//2, w - w//2: 2*w - w//2]).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapz = torch.from_numpy(self.offsetMapz[l - l//2: 2*l - l//2, h - h//2: 2*h - h//2, w - w//2: 2*w - w//2]).cuda(self.use_gpu).float() / rr

        self.offsetMapx = torch.from_numpy(
            self.offsetMapx[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float() / 63
        self.offsetMapy = torch.from_numpy(
            self.offsetMapy[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float() / 63
        self.offsetMapz = torch.from_numpy(
            self.offsetMapz[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float() / 63

    def forward(self, featureMaps, landmarks):

        imageNum, landmarkNum, l, h, w = featureMaps.size()
        landmarkNum = int(landmarkNum / 4)
        # ~ print ("size: ", featureMaps[0].size())
        # ~ print ()
        lossOff = 0
        lossReg = 0
        loss = 0

        indexs = self.binary_class_groundTruth > 0
        # print("3")
        temloss = [[self.binaryLoss(featureMaps[imageId][landmarkId * 4], self.binary_class_groundTruth), \
 \
                    self.huberLoss(featureMaps[imageId][landmarkId * 4 + 1], \
                                   self.offsetMapx), \
 \
                    self.huberLoss(featureMaps[imageId][landmarkId * 4 + 2], \
                                   self.offsetMapy), \
 \
                    self.huberLoss(featureMaps[imageId][landmarkId * 4 + 3], \
                                   self.offsetMapz)]

                   for imageId in range(imageNum)
                   for landmarkId in range(landmarkNum)]
        loss2 = (sum([sum(temloss[ind]) for ind in range(imageNum * landmarkNum)])) / (imageNum * landmarkNum)
        # print("4")

        return loss2

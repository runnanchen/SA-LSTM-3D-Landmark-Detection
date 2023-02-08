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

def analysis_result(landmarkNum, Off):
    SDR = np.zeros((landmarkNum, 8))
    SD = np.zeros((landmarkNum))
    MRE = np.mean(Off, axis=0)

    for landmarkId in range(landmarkNum):
        landmarkCol = Off[:, landmarkId]
        # print (np.max(landmarkCol))
        test_coarse_mm = np.array([landmarkCol[landmarkCol <= 1].size,
                                   landmarkCol[landmarkCol <= 2].size,
                                   landmarkCol[landmarkCol <= 3].size,
                                   landmarkCol[landmarkCol <= 4].size,
                                   landmarkCol[landmarkCol <= 5].size,
                                   landmarkCol[landmarkCol <= 6].size,
                                   landmarkCol[landmarkCol <= 7].size,
                                   # landmarkCol[landmarkCol <= 8].size,
                                   # landmarkCol[landmarkCol <= 4].size,
                                   landmarkCol[landmarkCol <= 8].size])
        SDR[landmarkId, :] = test_coarse_mm / landmarkCol.shape[0]
        SD[landmarkId] = np.sqrt(
            np.sum(np.power(landmarkCol - MRE[landmarkId], 2)) / (landmarkCol.shape[0] - 1))

    return SDR, SD, MRE

def adjustment(ROIs, labels):
    temoff = (ROIs - labels)
    temoff[temoff > 0.055] = temoff[temoff > 0.055] * 0 + 0.055
    temoff[temoff < -0.055] = temoff[temoff < -0.055] * 0 - 0.055
    ROIs = labels + temoff
    return ROIs

def Mydist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def Mydist3D(a, b):
    z1, x1, y1 = a
    z2, x2, y2 = b
    return math.sqrt((z2 - z1) ** 2 + (x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_coordinates_from_coarse_heatmaps(predicted_heatmap, global_coordinate):
    lent = len(predicted_heatmap)
    index = [1, 2, 0]
    global_coordinate_permute = global_coordinate.permute(3, 0, 1, 2)
    predict = [torch.sum((global_coordinate_permute * predicted_heatmap[i]).view(3, -1), dim = 1).unsqueeze(0) for i in range(lent)]
    predict = torch.cat(predict, dim=0)
    return predict[:, index]

def get_coordinates_from_fine_heatmaps(heatMaps, global_coordinate):
    lent = len(heatMaps)
    global_heatmap = [torch.sigmoid(heatMaps[i]) for i in range(lent)]
    global_heatmap = [global_heatmap[i] / global_heatmap[i].sum() for i in range(lent)]
    index = [1, 2, 0]
    global_coordinate_permute = global_coordinate.permute(3, 0, 1, 2)
    predict = [torch.sum((global_coordinate_permute * global_heatmap[i]).view(3, -1), dim = 1).unsqueeze(0) for i in range(lent)]
    predict = torch.cat(predict, dim=0)
    return predict[:, index]

def get_fine_errors(predicted_offset, labels, size_tensor):
    # take the last prediction as the final prediction
    predict = predicted_offset[-1, :, :] * size_tensor
    labels_b = labels * size_tensor
    # 0.3 is the spacing per voxel
    tem_dist = torch.sqrt(torch.sum(torch.pow(predict.squeeze() - labels_b.squeeze(), 2), 1)).unsqueeze(0) * 0.3
    return tem_dist

def get_coarse_errors(coarse_landmarks, global_coordinate, labels, size_tensor):
    predict = coarse_landmarks * size_tensor
    labels_b = labels * size_tensor
    # 0.3 is the spacing per voxel
    tem_dist = torch.sqrt(torch.sum(torch.pow(predict.squeeze() - labels_b.squeeze(), 2), 1)).unsqueeze(0) * 0.3
    return tem_dist

def get_global_feature(ROIs, coarse_feature, landmarkNum):
    X1, Y1, Z1 = ROIs[:, :, 0], ROIs[:, :, 1], ROIs[:, :, 2]
    L, H, W = coarse_feature.size()[-3:]
    X1, Y1, Z1 = np.round(X1 * (H - 1)).astype("int"), np.round(Y1 * (W - 1)).astype("int"), np.round(Z1 * (L - 1)).astype("int")
    global_embedding = torch.cat([coarse_feature[:, :, Z1[0, i], X1[0, i], Y1[0, i]] for i in range(landmarkNum)], dim=0).unsqueeze(0)
    return global_embedding

def getcropedInputs_related(ROIs, labels, inputs_origin, useGPU, index, config):
    labels_b = labels.detach().cpu().numpy()
    landmarks = ROIs
    landmarkNum = len(inputs_origin)

    b, c, l, h, w = inputs_origin[0].size()

    L, H, W = config.origin_image_size
    cropSize = 0
    if index == 0:
        cropSize = 32
    elif index == 1:
        cropSize = 16
    else:
        cropSize = 8

    # ~ print ("origin ", inputs_origin.size())

    X1, Y1, Z1 = landmarks[:, :, 0], landmarks[:, :, 1], landmarks[:, :, 2]
    X1, Y1, Z1 = np.round(X1 * (H - 1)).astype("int"), np.round(Y1 * (W - 1)).astype("int"), np.round(Z1 * (L - 1)).astype("int")

    X2, Y2, Z2 = labels_b[:, :, 0], labels_b[:, :, 1], labels_b[:, :, 2]
    X2, Y2, Z2 = np.round(X2 * (H - 1)).astype("int"), np.round(Y2 * (W - 1)).astype("int"), np.round(Z2 * (L - 1)).astype("int")

    X, Y, Z = X1 - X2 + int(h/2), Y1 - Y2 + int(w/2), Z1 - Z2 + int(l/2)
    # print(X, Y, Z)


    cropedDICOMs = []
    flag = True
    for landmarkId in range(landmarkNum):
        z, x, y = Z[0][landmarkId], X[0][landmarkId], Y[0][landmarkId]

        # if z<0 or z >= l or x < 0 or x >=h or y < 0 or y >= w:
        #     cropedDICOMs.append(torch.zeros(1, 1, 32, 32, 32))
        #     continue

        lz, uz, lx, ux, ly, uy = z - cropSize, z + cropSize, x - cropSize, x + cropSize, y - cropSize, y + cropSize
        lzz, uzz, lxx, uxx, lyy, uyy = max(lz, 0), min(uz, l), max(lx, 0), min(ux, h), max(ly, 0), min(uy, w)

        # ~ print (z, x, y)
        # ~ print ("boxes ", lz, uz, lx, ux, ly, uy)
        cropedDICOM = inputs_origin[landmarkId][:, :, lzz: uzz, lxx: uxx, lyy: uyy].clone()
        # ~ print ("check before", cropedDICOM.size())
        if lz < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, 0 - lz, curentX, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 2)
        if uz > l:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, uz - l, curentX, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 2)
        if lx < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, 0 - lx, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 3)
        if ux > h:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, ux - h, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 3)
        if ly < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, curentX, 0 - ly)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 4)
        if uy > w:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, curentX, uy - w)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 4)

        # cropedDICOMs.append(cropedDICOM)
        cropedDICOMs.append(F.upsample(cropedDICOM, size=(32, 32, 32), mode='trilinear'))

    # ~ print (cropedDICOMs.size())
    return cropedDICOMs

def getcropedInputs(ROIs, inputs_origin, cropSize, useGPU):
    landmarks = ROIs
    landmarkNum = landmarks.shape[1]
    b, c, l, h, w = inputs_origin.size()

    # l, h, w = 576, 768, 768
    cropSize = int(cropSize / 2)
    # ~ print ("origin ", inputs_origin.size())
    X, Y, Z = landmarks[:, :, 0], landmarks[:, :, 1], landmarks[:, :, 2]
    X, Y, Z = np.round(X * (h - 1)).astype("int"), np.round(Y * (w - 1)).astype("int"), np.round(Z * (l - 1)).astype(
        "int")
    cropedDICOMs = []
    flag = True
    for landmarkId in range(landmarkNum):

        z, x, y = Z[0][landmarkId], X[0][landmarkId], Y[0][landmarkId]
        lz, uz, lx, ux, ly, uy = z - cropSize, z + cropSize, x - cropSize, x + cropSize, y - cropSize, y + cropSize
        lzz, uzz, lxx, uxx, lyy, uyy = max(lz, 0), min(uz, l), max(lx, 0), min(ux, h), max(ly, 0), min(uy, w)

        # ~ print (z, x, y)
        # ~ print ("boxes ", lz, uz, lx, ux, ly, uy)
        cropedDICOM = inputs_origin[:, :, lzz: uzz, lxx: uxx, lyy: uyy]
        # ~ print ("check before", cropedDICOM.size())
        if lz < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, 0 - lz, curentX, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 2)
        if uz > l:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, uz - l, curentX, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 2)
        if lx < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, 0 - lx, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 3)
        if ux > h:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, ux - h, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 3)
        if ly < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            # import pdb
            # pdb.set_trace()
            temTensor = torch.zeros(b, c, curentZ, curentX, 0 - ly)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 4)
        if uy > w:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, curentX, uy - w)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 4)

        cropedDICOMs.append(cropedDICOM)

    # ~ print (cropedDICOMs.size())
    return cropedDICOMs

def get_local_patches(ROIs, cropedtems, base_coordinate, usegpu):
    local_coordinate = []
    local_patches = []
    for i in range(len(cropedtems)):
        centre = torch.from_numpy(ROIs[0, i, :]).cuda(usegpu)
        tem = base_coordinate + centre
        local_coordinate.append(tem)
    local_patches = cropedtems
    return local_patches, local_coordinate

def getCroped(ROIs, outputs):
    # imageNum, landmarkNum * channels, Long, Height, Width
    Y1, Y2, Y3 = outputs[1], outputs[2], outputs[3]
    size1, size2, size3 = Y1.size()[2:], Y2.size()[2:], Y3.size()[2:]
    print(size1, size2, size3)

def resizeDICOM(DICOM, shape_DICOM):
    l, h, w = DICOM.shape[:3]
    newl, newh, neww = shape_DICOM
    scalel, scaleh, scalew = newl / l, newh / h, neww / w

    newDICOM = zoom(DICOM, (scalel, scaleh, scalew))
    print(newDICOM.shape)
    return newDICOM

def showDICOM(DICOM, label, predict, epoch, lent):

    # import pdb
    # pdb.set_trace()

    x, y, z = int(label[0] * 767), int(label[1] * 767), int(label[2] * 575)
    xx, yy, zz = int(predict[0] * 767), int(predict[1] * 767), int(predict[2] * 575)

    imageX = DICOM[:, x, :]
    imageY = DICOM[:, :, y]
    imageZ = DICOM[z, :, :]
    # ~ print (x, y, z)
    # ~ print ("imageX", imageX.shape)
    # ~ print ("imageY", imageY.shape)
    # ~ print ("imageZ", imageZ.shape)

    minvX, maxvX = np.min(imageX), np.max(imageX)
    minvY, maxvY = np.min(imageY), np.max(imageY)
    minvZ, maxvZ = np.min(imageZ), np.max(imageZ)

    imageX = (imageX - minvX) / (maxvX - minvX) * 255
    imageY = (imageY - minvY) / (maxvY - minvY) * 255
    imageZ = (imageZ - minvZ) / (maxvZ - minvZ) * 255

    imageX = Image.fromarray(imageX.astype('uint8'))
    imageX = imageX.convert('RGB')
    drawX = ImageDraw.Draw(imageX)

    imageY = Image.fromarray(imageY.astype('uint8'))
    imageY = imageY.convert('RGB')
    drawY = ImageDraw.Draw(imageY)

    imageZ = Image.fromarray(imageZ.astype('uint8'))



    imageZ = imageZ.convert('RGB')
    drawZ = ImageDraw.Draw(imageZ)
    r = int(DICOM.shape[0] / 80)


    positionX = (y - r, z - r, y + r, z + r)
    positionY = (x - r, z - r, x + r, z + r)
    positionZ = (y - r, x - r, y + r, x + r)

    positionXX = (yy - r, zz - r, yy + r, zz + r)
    positionYY = (xx - r, zz - r, xx + r, zz + r)
    positionZZ = (yy - r, xx - r, yy + r, xx + r)

    drawX.ellipse(positionXX, fill=(255, 0, 0))
    drawY.ellipse(positionYY, fill=(255, 0, 0))
    drawZ.ellipse(positionZZ, fill=(255, 0, 0))

    drawX.ellipse(positionX, fill=(0, 255, 0))
    drawY.ellipse(positionY, fill=(0, 255, 0))
    drawZ.ellipse(positionZ, fill=(0, 255, 0))


    imageX.save("vis_images/" + str(lent) + "_" + str(epoch) + "_imageX.jpg")
    imageY.save("vis_images/" + str(lent) + "_" + str(epoch) + "_imageY.jpg")
    imageZ.save("vis_images/" + str(lent) + "_" + str(epoch) + "_imageZ.jpg")

    # plt.suptitle("multi_image")
    # plt.subplot(1, 3, 1), plt.title("x")
    # plt.imshow(imageX, cmap='gray', interpolation='nearest'), plt.axis("off")
    # plt.subplot(1, 3, 2), plt.title("y")
    # plt.imshow(imageY, cmap='gray', interpolation='nearest'), plt.axis("off")
    # plt.subplot(1, 3, 3), plt.title("z")
    # plt.imshow(imageZ, cmap='gray', interpolation='nearest')
    # plt.savefig("filename.png")
    # print("filename.png")
    # plt.show()
    # dfdf = input()

def drawImage(image, coordindates_before, coordindates_after):
    # image = image_before
    # image = Image.fromarray((image * 255).astype('uint8'))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 15)
    t = 0

    for ide in range(68):
        r = 6
        t = t + 1
        # draw.rectangle(coordindates_before, outline = "red")
        # x, y = coordindates_after[ide]['x'], coordindates_after[ide]['y']
        x, y = coordindates_after[ide][0], coordindates_after[ide][1]
        position = (x - r, y - r, x + r, y + r)

        # draw.ellipse(position,fill = (0, 255, 0))

        draw.text((x, y), str(t), fill=(0, 255, 255), font=font)

    plt.imshow(image, cmap='gray', interpolation='nearest')
    image.save("compare.png")
    fdf = input()


# return image
def Mydist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def Mydist3D(a, b):
    z1, x1, y1 = a
    z2, x2, y2 = b
    return math.sqrt((z2 - z1) ** 2 + (x2 - x1) ** 2 + (y2 - y1) ** 2)

def getCoordinate_new(featureMaps, outputs2, lables, R1, R2, gpu, lastResult, coordinatesFine, config):
    imageNum, featureNum, l, h, w = featureMaps[0].size()
    _, _, l_2, h_2, w_2 = outputs2.size()
    landmarkNum = int(featureNum)
    corse_landmark = lastResult.detach().cpu().numpy()
    fine_landmark = coordinatesFine.detach().cpu().numpy()

    X1, Y1, Z1 = np.round(corse_landmark[:, :, 0] * 767).astype('int'), np.round(corse_landmark[:, :, 1] * 767).astype(
        'int'), np.round(corse_landmark[:, :, 2] * 575).astype('int')

    X2, Y2, Z2 = np.round(fine_landmark[:, :, 0] * h_2).astype('int'), np.round(fine_landmark[:, :, 1] * w_2).astype(
        'int'), np.round(fine_landmark[:, :, 2] * l_2).astype('int')

    X_off, Y_off, Z_off = X2 - h_2 // 2, Y2 - w_2 // 2, Z2 - l_2 // 2

    GX, GY, GZ = np.round(lables[:, :, 0].numpy() * 767).astype('int'), np.round(lables[:, :, 1].numpy() * 767).astype(
        'int'), np.round(lables[:, :, 2].numpy() * 575).astype('int')

    tot = np.zeros((imageNum, landmarkNum))
    for imageId in range(imageNum):
        for landmarkId in range(landmarkNum):
            x, y, z = X1[imageId][landmarkId], Y1[imageId][landmarkId], Z1[imageId][landmarkId]

            x_off, y_off, z_off = X_off[imageId][landmarkId], Y_off[imageId][landmarkId], Z_off[imageId][landmarkId]

            x_2 = x_off + x
            y_2 = y_off + y
            z_2 = z_off + z

            xx, yy, zz = GX[imageId][landmarkId], GY[imageId][landmarkId], GZ[imageId][landmarkId]
            # tem_dist = Mydist3D((0, 0, 0), (x_off, y_off, z_off))
            # tem_dist1 = Mydist3D((z, x, y), (zz, xx, yy))
            tem_dist2 = Mydist3D((z_2, x_2, y_2), (zz, xx, yy))
            tot[imageId][landmarkId] = tem_dist2

    return (tot)
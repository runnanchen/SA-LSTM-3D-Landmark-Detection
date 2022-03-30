from __future__ import print_function, division
import torch
import numpy as np
import math

def Mydist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def Mydist3D(a, b):
    z1, x1, y1 = a
    z2, x2, y2 = b
    return math.sqrt((z2 - z1) ** 2 + (x2 - x1) ** 2 + (y2 - y1) ** 2)

def analysis_result(landmarkNum, Off):
    SDR = np.zeros((landmarkNum, 5))
    SD = np.zeros((landmarkNum))
    MRE = np.mean(Off, axis=0)

    for landmarkId in range(landmarkNum):
        landmarkCol = Off[:, landmarkId]
        # print (np.max(landmarkCol))
        test_coarse_mm = np.array([landmarkCol[landmarkCol <= 2].size,
                                   landmarkCol[landmarkCol <= 2.5].size,
                                   landmarkCol[landmarkCol <= 3].size,
                                   landmarkCol[landmarkCol <= 4].size,
                                   landmarkCol[landmarkCol <= 8].size])
        SDR[landmarkId, :] = test_coarse_mm / landmarkCol.shape[0]
        SD[landmarkId] = np.sqrt(
            np.sum(np.power(landmarkCol - MRE[landmarkId], 2)) / (landmarkCol.shape[0] - 1))

    return SDR, SD, MRE

def get_coordinates_from_coarse_heatmaps(predicted_heatmap, global_coordinate):
    lent = predicted_heatmap.size()[0]
    index = [1, 2, 0]
    global_coordinate_permute = global_coordinate.permute(3, 0, 1, 2)
    predict = [torch.sum((global_coordinate_permute * predicted_heatmap[i]).view(3, -1), dim = 1).unsqueeze(0) for i in range(lent)]
    predict = torch.cat(predict, dim=0)
    return predict[:, index]

def get_fine_errors(predicted_offset, labels, size_tensor):

    predict = predicted_offset[-1, :, :] * size_tensor
    labels_b = labels * size_tensor
    tem_dist = torch.sqrt(torch.sum(torch.pow(predict.squeeze() - labels_b.squeeze(), 2), 1)).unsqueeze(0) * 0.3

    return tem_dist


def get_coarse_errors(coordinates1, lables):
    coordinates1_b = coordinates1.clone()
    lables_b = lables.clone()

    coordinates1_b[:, :, 0] = coordinates1_b[:, :, 0] * (768 - 1)
    coordinates1_b[:, :, 1] = coordinates1_b[:, :, 1] * (768 - 1)
    coordinates1_b[:, :, 2] = coordinates1_b[:, :, 2] * (576 - 1)

    lables_b[:, :, 0] = lables_b[:, :, 0] * (768 - 1)
    lables_b[:, :, 1] = lables_b[:, :, 1] * (768 - 1)
    lables_b[:, :, 2] = lables_b[:, :, 2] * (576 - 1)

    tem_dist = torch.sqrt(torch.sum(torch.pow(coordinates1_b - lables_b, 2), 2)) * 0.3

    return tem_dist

def get_global_feature(ROIs, coarse_feature, landmarkNum):
    X1, Y1, Z1 = ROIs[:, :, 0], ROIs[:, :, 1], ROIs[:, :, 2]
    L, H, W = coarse_feature.size()[-3:]
    X1, Y1, Z1 = np.round(X1 * (H - 1)).astype("int"), np.round(Y1 * (W - 1)).astype("int"), np.round(Z1 * (L - 1)).astype("int")
    global_embedding = torch.cat([coarse_feature[:, :, Z1[0, i], X1[0, i], Y1[0, i]] for i in range(landmarkNum)], dim=0).unsqueeze(0)
    return global_embedding

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

# return image

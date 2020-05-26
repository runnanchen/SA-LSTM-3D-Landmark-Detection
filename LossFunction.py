from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import math
import MyUtils

class coarse_heatmap(nn.Module):
    def __init__(self, use_gpu, batchSize, landmarkNum, image_scale):
        super(coarse_heatmap, self).__init__()
        self.use_gpu = use_gpu
        self.batchSize = batchSize
        self.landmarkNum = landmarkNum
        self.l1Loss = nn.L1Loss(size_average=False)
        self.Long, self.higth, self.width = image_scale
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

    def forward(self, predicted_heatmap, local_coordinate, labels, size_tensor, phase):
        loss = 0
        labels_b = labels * torch.tensor([self.higth - 1, self.width - 1, self.Long - 1]).cuda(self.use_gpu)
        labels_b = np.round(labels_b.detach().cpu().numpy()).astype("int")
        X, Y, Z = labels_b[0, :, 0], labels_b[0, :, 1], labels_b[0, :, 2]

        for i in range(self.landmarkNum):
            coarse_heatmap = self.HeatMap_groundTruth[self.Long - Z[i]: 2 * self.Long - Z[i],
                                                                        self.higth - X[i]: 2 * self.higth - X[i],
                                                                        self.width - Y[i]: 2 * self.width - Y[i]]

            loss += torch.abs(predicted_heatmap[i] - coarse_heatmap / coarse_heatmap.sum()).sum()
        return loss

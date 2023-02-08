from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
import MyUtils
import torch.nn.functional as F
import MyNetworkLayer as MNL
# (16, 11) for 600; (10, 8) for 300; (37, 30) for 1000
import math


class Unet(nn.Module):
    def __init__(self, landmarksNum, useGPU):
        super(Unet, self).__init__()
        ndf = 64
        # ~ conNum = 64
        self.unet = MNL.U_Net3D(ndf, ndf)

        self.landmarkNum = landmarksNum
        self.useGPU = useGPU
        self.conv3d_L0 = nn.Conv3d(ndf, landmarksNum, 3, 1, 1)

    def forward(self, x):
        x = self.unet(x)
        localAppear = self.conv3d_L0(x)

        # spatialConfigure = self.Upsample4(self.conv3d_SC(self.maxPooling4(localAppear)))
        # heatMaps = F.sigmoid(localAppear * spatialConfigure)
        heatMaps = F.sigmoid(localAppear)

        heatmap_sum = torch.sum(heatMaps.view(self.landmarkNum, -1), dim=1)

        global_heatmap = [heatMaps[0, i, :, :, :].squeeze() / heatmap_sum[i] for i in range(self.landmarkNum)]

        return global_heatmap, global_heatmap

        return heatMaps, heatMaps

class scn3d(nn.Module):
    def __init__(self, landmarksNum, useGPU):
        super(scn3d, self).__init__()
        ndf = 64
        # ~ conNum = 64
        self.conv3d_M1 = nn.Sequential(
            # 144 192 192
            nn.Conv3d(1, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),

            # nn.Conv3d(ndf, ndf, 3, 1, 1),
            # nn.BatchNorm3d(ndf,track_running_stats=False),
            # nn.ReLU(True),

            # ~ nn.Conv3d(ndf, conNum, 1, 1, 0, bias=False),
        )
        self.conv3d_M2 = nn.Sequential(
            # 72 96 96
            nn.MaxPool3d(2, 2),
            nn.Conv3d(ndf, ndf, 3, 1, 1),
            # nn.Conv3d(ndf, ndf, 4, 2, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),

        )
        '''
        self.conv3d_M3 = nn.Sequential(
            # 36 48 48
            nn.MaxPool3d(2, 2),
            nn.Conv3d(ndf, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf,track_running_stats=False),
            nn.ReLU(True),

        )
        '''
        self.conv3d_bottom = nn.Sequential(
            # 24 24 24

            nn.MaxPool3d(2, 2),
            # nn.ConvTranspose3d(ndf, ndf, 3, 1, 1),
            nn.Conv3d(ndf, ndf, 3, 1, 1),
            # nn.Conv3d(ndf, ndf, 4, 2, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),

            # nn.ConvTranspose3d(ndf, ndf, 3, 1, 1),
            nn.Conv3d(ndf, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),

        )

        self.conv3d_L0 = nn.Conv3d(ndf, landmarksNum, 3, 1, 1)

        self.conv3d_L1 = nn.Sequential(
            # 144 192 192

            nn.Conv3d(ndf, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),

            # ~ nn.Conv3d(ndf, conNum, 1, 1, 0, bias=False),
        )
        self.conv3d_L2 = nn.Sequential(
            # 72 96 96
            nn.Conv3d(ndf, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),
        )
        self.conv3d_L3 = nn.Sequential(
            # 36 48 48
            nn.Conv3d(ndf, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),
        )

        self.conv3d_SC = nn.Sequential(
            # 36 48 48
            nn.Conv3d(landmarksNum, ndf, 7, 1, 3),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv3d(ndf, ndf, 7, 1, 3),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv3d(ndf, landmarksNum, 7, 1, 3),
            nn.Tanh()
        )

        self.maxPooling4 = nn.MaxPool3d(4, 4)
        self.Upsample4 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.Upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.landmarkNum = landmarksNum
        self.useGPU = useGPU

    def forward(self, x):
        x1 = self.conv3d_M1(x)
        x2 = self.conv3d_M2(x1)
        # x3 = self.conv3d_M3(x2)
        x = self.conv3d_bottom(x2)
        # x = self.Upsample2(x) + self.conv3d_L3(x3)
        x = self.Upsample2(x) + self.conv3d_L2(x2)
        x = self.Upsample2(x) + self.conv3d_L1(x1)
        localAppear = self.conv3d_L0(x)

        # spatialConfigure = self.Upsample4(self.conv3d_SC(self.maxPooling4(localAppear)))
        # heatMaps = F.sigmoid(localAppear * spatialConfigure)
        heatMaps = F.sigmoid(localAppear)

        heatmap_sum = torch.sum(heatMaps.view(self.landmarkNum, -1), dim=1)

        global_heatmap = [heatMaps[0, i, :, :, :].squeeze() / heatmap_sum[i] for i in range(self.landmarkNum)]

        return global_heatmap, global_heatmap

        return heatMaps, heatMaps


class coarseNet(nn.Module):
    def __init__(self, config):
        # landmarkNum, use_gpu, image_scale
        super(coarseNet, self).__init__()
        self.landmarkNum = config.landmarkNum
        self.usegpu = config.use_gpu
        self.image_scale = config.image_scale
        self.u_net = MNL.U_Net3D(1, 64)
        self.conv3d = nn.Sequential(
            nn.Conv3d(64, config.landmarkNum, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        global_features = self.u_net(x)
        x = self.conv3d(global_features) + 1e-9
        heatmap_sum = torch.sum(x.view(self.landmarkNum, -1), dim=1)
        # print(heatmap_sum.size())
        global_heatmap = [x[0, i, :, :, :].squeeze() / heatmap_sum[i] for i in range(self.landmarkNum)]

        return global_heatmap, global_features

class fine_LSTM(nn.Module):
    def __init__(self, config):
        super(fine_LSTM, self).__init__()

        # landmarkNum, use_gpu, iteration, cropSize

        self.landmarkNum = config.landmarkNum
        self.usegpu = config.use_gpu
        self.encoder = MNL.U_Net3D_encoder(1, 64)
        self.iteration = config.iteration
        self.crop_size = config.crop_size
        self.origin_image_size = config.origin_image_size
        self.config = config

        w, h, l = self.origin_image_size
        # (576, 768, 768)

        self.size_tensor = torch.tensor([1 / (l - 1), 1 / (h - 1), 1 / (w - 1)]).cuda(self.usegpu)

        self.decoders_offset_x = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + 64, 1, 0, groups=self.landmarkNum)
        self.decoders_offset_y = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + 64, 1, 0, groups=self.landmarkNum)
        self.decoders_offset_z = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + 64, 1, 0, groups=self.landmarkNum)


        self.attention_gate_share = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.Tanh(),
            # nn.Linear(256, 1)
            # nn.Conv1d(landmarkNum, landmarkNum, 256, 1, 0, groups=landmarkNum),
        )
        self.attention_gate_head = nn.Conv1d(self.landmarkNum, self.landmarkNum, 256, 1, 0, groups=self.landmarkNum)
        self.graph_attention = MNL.graph_attention(64, self.usegpu)

    def forward(self, coarse_landmarks, labels, inputs_origin, coarse_feature, phase, size_tensor_inv):

        # cropedtems = MyUtils.getcropedInputs_related(ROIs, labels, inputs_origin, -1, 0)
        # cropedtems = torch.cat([cropedtems[i].cuda(self.usegpu) for i in range(len(cropedtems))], dim=0)
        # features = self.encoder(cropedtems).squeeze().unsqueeze(0)
        # global_feature = MyUtils.get_global_feature(ROIs, coarse_feature)
        # global_feature = self.graph_attention(ROIs, global_feature)
        # features = torch.cat((features, global_feature),dim=2)
        # x, y, z = self.decoders_offset_x(features), self.decoders_offset_y(features), self.decoders_offset_z(features)
        # predict = torch.cat([x, y, z], dim=2) * self.size_tensor.cuda(self.usegpu) + torch.from_numpy(ROIs).cuda(self.usegpu)

        h_state = 0
        predicts = []
        c_state = 0
        predict = coarse_landmarks.detach()

        for i in range(0, self.iteration):
            ROIs = 0
            if phase == 'train':
                if i == 0:
                    ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=32.0 / self.origin_image_size[2] / 3, size = labels.size())).cuda(self.usegpu).float()
                elif i == 1:
                    ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=16.0 / self.origin_image_size[2] / 3, size = labels.size())).cuda(self.usegpu).float()
                else:
                    ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=8.0 / self.origin_image_size[2] / 3, size = labels.size())).cuda(self.usegpu).float()
            else:
                ROIs = predict

            ROIs = MyUtils.adjustment(ROIs, labels)


            cropedtems = MyUtils.getcropedInputs_related(ROIs.detach().cpu().numpy(), labels, inputs_origin, -1, i, self.config)
            cropedtems = torch.cat([cropedtems[i].cuda(self.usegpu) for i in range(len(cropedtems))], dim=0)
            features = self.encoder(cropedtems).squeeze().unsqueeze(0)

            global_feature = MyUtils.get_global_feature(ROIs.detach().cpu().numpy(), coarse_feature, self.landmarkNum)
            global_feature = self.graph_attention(ROIs, global_feature)
            features = torch.cat((features, global_feature), dim=2)
            # features = self.graph_attention(ROIs, features)

            # h_state = features
            # c_state = ROIs
            if i == 0:
                h_state = features
                c_state = ROIs
            else:
                gate_f = self.attention_gate_head(self.attention_gate_share(h_state.squeeze()).unsqueeze(0))
                gate_a = self.attention_gate_head(self.attention_gate_share(features.squeeze()).unsqueeze(0))
                gate = torch.softmax(torch.cat([gate_f, gate_a], dim=2), dim=2)

                h_state = h_state * gate[0, :, 0].view(1, -1, 1) + features * gate[0, :, 1].view(1, -1, 1)
                c_state = c_state * gate[0, :, 0].view(1, -1, 1) + ROIs * gate[0, :, 1].view(1, -1, 1)
                # c_state = ROIs

            x, y, z = self.decoders_offset_x(h_state), self.decoders_offset_y(h_state), self.decoders_offset_z(h_state)
            # print(size_tensor_inv)
            predict = torch.cat([x, y, z], dim=2) * size_tensor_inv + c_state
            predicts.append(predict.float())

        predicts = torch.cat(predicts, dim=0)

        return predicts

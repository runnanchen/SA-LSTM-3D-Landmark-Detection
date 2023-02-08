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
# (16, 11) for 600; (10, 8) for 300; (37, 30) for 1000

class graph_attention(nn.Module):
    def __init__(self, feature_size, usegpu):
        super(graph_attention, self).__init__()
        c = feature_size
        self.c = c
        self.usegpu = usegpu
        self.i = nn.Sequential(
            nn.Linear(c+3, c//2),
        )
        self.j = nn.Sequential(
            nn.Linear(c+3, c//2),
        )
        self.k = nn.Sequential(
            nn.Linear(c+3, c//2),
        )
        self.restore = nn.Sequential(
            nn.Linear(c//2, c),
        )


    def forward(self, ROIs, features):
        input_size = features.size()
        features_concat = torch.cat((ROIs, features), dim=2).squeeze()
        fi = self.i(features_concat)
        fj = self.j(features_concat).permute(1, 0)
        fk = self.k(features_concat)

        attention_ij = torch.sigmoid(torch.matmul(fi, fj))
        attention_sum = torch.sum(attention_ij, 1).view(-1, 1)
        attention_ij = attention_ij/attention_sum
        # attention_ij = torch.softmax(attention_ij, dim=1)
        features = features + self.restore(torch.matmul(attention_ij, fk))

        return features.view(input_size)


class non_local_bb(nn.Module):
    def __init__(self, feature_size, use_gpu):
        super(non_local_bb, self).__init__()
        c = feature_size
        self.eyes = torch.eye(18*24*24).cuda(use_gpu)
        self.i = nn.Sequential(
            nn.Conv3d(c+3, c//2, 1, 1, 0),
            # nn.BatchNorm3d(c // 2, track_running_stats=False),
            # nn.Tanh(),
        )
        self.j = nn.Sequential(
            nn.Conv3d(c+3, c//2, 1, 1, 0),
            # nn.BatchNorm3d(c // 2, track_running_stats=False),
            # nn.Tanh()
        )
        self.k = nn.Sequential(
            nn.Conv3d(c+3, c//2, 1, 1, 0),
            # nn.BatchNorm3d(c//2, track_running_stats=False),
            # nn.ReLU(),
        )
        self.c = c
        self.restore = nn.Sequential(
            nn.Conv3d(c//2, c, 1, 1, 0),
            # nn.BatchNorm3d(c, track_running_stats=False),
            # nn.ReLU(),
        )

        self.global_filter = nn.Sequential(
            nn.Conv3d(c, 1, 1, 1, 0),
            # nn.BatchNorm3d(c, track_running_stats=False),
            nn.ReLU(),
        )

        gl, gh, gw = 18, 24, 24
        global_coordinate = torch.ones(gl, gh, gw, 3).float()
        for i in range(gl):
            global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
        for i in range(gh):
            global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
        for i in range(gw):
            global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
        global_coordinate = global_coordinate.cuda(use_gpu) * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)]).cuda(use_gpu)
        self.global_coordinate = global_coordinate.permute(3, 0, 1, 2).unsqueeze(0) - 0.5

    def forward(self, features):
        input_size = features.size()
        features_concat = torch.cat((self.global_coordinate, features), dim=1)
        # filter_feature = self.global_filter(features)
        fi = self.i(features_concat).squeeze().permute(1, 2, 3, 0).view(-1, self.c//2)
        fj = self.j(features_concat).squeeze().view(self.c//2, -1)
        fk = self.k(features_concat)
        mid_size = fk.size()
        fk = fk.squeeze().permute(1, 2, 3, 0).view(-1, self.c // 2)

        attention_ij = torch.matmul(fi, fj)
        # attention_ij = torch.softmax(attention_ij, dim=1)
        attention_ij = attention_ij / 10368
        features = features + self.restore(torch.matmul(attention_ij, fk).permute(1, 0).view(mid_size))
        # features = features + torch.matmul(attention_ij, fk).permute(1, 0).view(input_size)

        return features


class non_local_b(nn.Module):
    def __init__(self, feature_size, use_gpu):
        super(non_local_b, self).__init__()
        c = feature_size
        self.eyes = torch.eye(18*24*24).cuda(use_gpu)
        self.i = nn.Sequential(
            nn.Conv3d(c, c//2, 1, 1, 0),
            # nn.BatchNorm3d(c // 2, track_running_stats=False),
            # nn.Tanh(),
        )
        self.j = nn.Sequential(
            nn.Conv3d(c, c//2, 1, 1, 0),
            # nn.BatchNorm3d(c // 2, track_running_stats=False),
            # nn.Tanh()
        )
        self.k = nn.Sequential(
            nn.Conv3d(c, c//2, 1, 1, 0),
            # nn.BatchNorm3d(c//2, track_running_stats=False),
            # nn.ReLU(),
        )
        self.c = c
        self.restore = nn.Sequential(
            nn.Conv3d(c//2, c, 1, 1, 0),
            # nn.BatchNorm3d(c, track_running_stats=False),
            # nn.ReLU(),
        )
        gl, gh, gw = 18, 24, 24
        global_coordinate = torch.ones(gl, gh, gw, 3).float()
        for i in range(gl):
            global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
        for i in range(gh):
            global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
        for i in range(gw):
            global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
        global_coordinate = global_coordinate.cuda(use_gpu) * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)]).cuda(use_gpu)
        self.global_coordinate = global_coordinate.permute(3, 0, 1, 2).unsqueeze(0) - 0.5

    def forward(self, features):
        input_size = features.size()
        # features_concat = torch.cat((self.global_coordinate, features), dim=1)
        fi = self.i(features).squeeze().permute(1, 2, 3, 0).view(-1, self.c//2)
        fj = self.j(features).squeeze().view(self.c//2, -1)
        # fk = self.k(features)
        # mid_size = fk.size()
        fk = features.squeeze().permute(1, 2, 3, 0).view(-1, self.c)

        attention_ij = torch.matmul(fi, fj)
        # attention_ij = torch.softmax(attention_ij, dim=1)
        attention_ij = attention_ij / 10368
        # features = features + self.restore(torch.matmul(attention_ij, fk).permute(1, 0).view(mid_size))
        features = features + torch.matmul(attention_ij, fk).permute(1, 0).view(input_size)

        return features


class non_local(nn.Module):
    def __init__(self, feature_size, use_gpu):
        super(non_local, self).__init__()
        c = feature_size
        self.head = 1
        self.hsize = c//4
        self.eyes = torch.eye(18*24*24).cuda(use_gpu)
        self.i = nn.Sequential(
            nn.Conv3d(c, self.hsize * self.head, 1, 1, 0),
            # nn.BatchNorm3d(c // 2, track_running_stats=False),
            # nn.Tanh(),
        )
        self.j = nn.Sequential(
            nn.Conv3d(c, self.hsize, 1, 1, 0),
            # nn.BatchNorm3d(c // 2, track_running_stats=False),
            # nn.Tanh()
        )
        self.k = nn.Sequential(
            nn.Conv3d(c, self.hsize, 1, 1, 0),
            # nn.BatchNorm3d(c//2, track_running_stats=False),
            # nn.ReLU(),
        )
        self.c = c
        self.restore = nn.Sequential(
            nn.Conv3d(self.hsize, c, 1, 1, 0),
            # nn.BatchNorm3d(c, track_running_stats=False),
            # nn.ReLU(),
        )
        # self.attention_agregate = nn.Conv2d(self.head, 1, 1, 1, 0)
        self.attention_agregate = nn.Parameter(torch.zeros(size=(self.head, 1)))
        nn.init.xavier_uniform_(self.attention_agregate.data, gain=1.414)

    def forward(self, features):
        input_size = features.size()
        fi = self.i(features).view(self.head, self.hsize, -1).permute(0, 2, 1)
        fj = self.j(features).view(self.hsize, -1)
        # fk = self.k(features)
        # mid_size = fk.size()
        fk = features.view(self.c, -1).permute(1, 0)

        # attention_ij = torch.matmul(fi, fj)
        attention_ij = torch.matmul(torch.matmul(fi, fj).permute(1, 2, 0), self.attention_agregate).squeeze()
        # print(attention_ij.size())
        # attention_ij = self.attention_agregate(attention_ij.unsqueeze(0)).squeeze()
        attention_ij = attention_ij / (input_size[2] * input_size[3] * input_size[4])
        # print(attention_ij.size())
        # features = features + self.restore(torch.matmul(attention_ij, fk).permute(1, 0).view(mid_size))
        features = features + torch.matmul(attention_ij, fk).permute(1, 0).view(input_size)

        return features

class Graph_attention3d_layer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, landmarksNum):
        super(Graph_attention3d_layer, self).__init__()
        self.landmarkNum = landmarksNum
        self.attention_share = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(3072, 256),
            nn.Tanh(),
            # nn.Linear(256, 1)
            # nn.Conv1d(landmarkNum, landmarkNum, 256, 1, 0, groups=landmarkNum),
        )
        self.conv3d = nn.Sequential(
            nn.BatchNorm3d(landmarksNum, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv3d(landmarksNum, landmarksNum, 3, 1, 1),
        )
        self.maxPooling6 = nn.MaxPool3d(6, 6)
        self.Fa = nn.Sequential(
            nn.Linear(2 * 256, 1),
            # nn.Sigmoid()
        )

    def forward(self, features):
        input_size = features.size()
        N = self.landmarkNum
        features = self.conv3d(features)
        attention = self.attention_share(self.maxPooling6(features).view(N, -1))
        # print(attention.size())
        attention = torch.cat([attention.repeat(1, N).view(N * N, -1), attention.repeat(N, 1)], dim=1)
        # print(attention.size())
        attention = self.Fa(attention).view(N, N) + 1e-9
        # print(attention.size())
        # attention_sum = torch.sum(attention, dim=1)
        attention = F.softmax(attention, dim=1)
        # attention = (attention.permute(1, 0) / attention_sum).permute(1, 0)

        # print(torch.sum(attention,dim=1))

        # print(attention)
        # features = torch.matmul(attention, features.view(N, -1)).view(input_size)
        # return h_prime
        return features


class U_Net3D(nn.Module):
    def __init__(self, fin, fout):
        super(U_Net3D, self).__init__()
        ndf = 32
        # ~ conNum = 64
        self.Lconv1 = nn.Sequential(
            nn.Conv3d(fin, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),

            # nn.Conv3d(ndf, ndf, 3, 1, 1),
            # nn.BatchNorm3d(ndf,track_running_stats=False),
            # nn.ReLU(True),
        )
        self.Lconv2 = nn.Sequential(

            # nn.MaxPool3d(2, 2),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm3d(ndf * 2, track_running_stats=False),
            nn.ReLU(True),

            # nn.Conv3d(ndf*2, ndf*2, 3, 1, 1),
            # nn.BatchNorm3d(ndf*2,track_running_stats=False),
            # nn.ReLU(True),
        )

        self.Lconv3 = nn.Sequential(

            #nn.MaxPool3d(2, 2),
            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm3d(ndf*4,track_running_stats=False),
            nn.ReLU(True),

            #nn.Conv3d(ndf*4, ndf*4, 3, 1, 1),
            #nn.BatchNorm3d(ndf*4,track_running_stats=False),
            #nn.ReLU(True),

        )


        self.Lconv4 = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            nn.Conv3d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm3d(ndf*8,track_running_stats=False),
            nn.ReLU(True),

            #nn.Conv3d(ndf*8, ndf*8, 3, 1, 1),
            #nn.BatchNorm3d(ndf*8,track_running_stats=False),
            #nn.ReLU(True),
        )

        self.bottom = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm3d(ndf * 8, track_running_stats=False),
            nn.ReLU(True),

            # nn.Conv3d(ndf*16, ndf*16, 3, 1, 1),
            # nn.BatchNorm3d(ndf*16,track_running_stats=False),
            # nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf * 8, ndf * 4, 1, 1, 0),
            nn.BatchNorm3d(ndf * 4, track_running_stats=False),
            nn.ReLU(True),
        )

        self.Rconv4 = nn.Sequential(
            nn.Conv3d(ndf*16, ndf*8, 3, 1, 1),
            nn.BatchNorm3d(ndf*8,track_running_stats=False),
            nn.ReLU(True),

            #nn.Conv3d(ndf*8, ndf*8, 3, 1, 1),
            #nn.BatchNorm3d(ndf*8,track_running_stats=False),
            #nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf*8, ndf*4, 1, 1, 0),
            nn.BatchNorm3d(ndf*4,track_running_stats=False),
            nn.ReLU(True),

        )

        self.Rconv3 = nn.Sequential(
            nn.Conv3d(ndf*8, ndf*4, 3, 1, 1),
            nn.BatchNorm3d(ndf*4,track_running_stats=False),
            nn.ReLU(True),

            #nn.Conv3d(ndf*4, ndf*4, 3, 1, 1),
            #nn.BatchNorm3d(ndf*4,track_running_stats=False),
            #nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf*4, ndf*2, 1, 1, 0),
            nn.BatchNorm3d(ndf*2,track_running_stats=False),
            nn.ReLU(True),
        )

        self.Rconv2 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 2, 3, 1, 1),
            nn.BatchNorm3d(ndf * 2, track_running_stats=False),
            nn.ReLU(True),

            # nn.Conv3d(ndf*2, ndf*2, 3, 1, 1),
            # nn.BatchNorm3d(ndf*2,track_running_stats=False),
            # nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf * 2, ndf, 1, 1, 0),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),
        )


        self.Rconv1 = nn.Sequential(
            nn.Conv3d(ndf * 2, fout, 3, 1, 1),
            nn.BatchNorm3d(fout, track_running_stats=False),
            nn.ReLU(True),

            # nn.Conv3d(ndf, ndf, 3, 1, 1),
            # nn.BatchNorm3d(ndf,track_running_stats=False),
            # nn.ReLU(True),

            # nn.Conv3d(ndf, fout, 3, 1, 1),
            # nn.BatchNorm3d(fout, track_running_stats=False),
            # nn.ReLU(True),
        )

    def forward(self, x):
        x1 = self.Lconv1(x)
        x2 = self.Lconv2(x1)
        x3 = self.Lconv3(x2)
        # x4 = self.Lconv4(x3)
        # ~ x5 = self.Lconv5(x4)
        x = torch.cat((self.bottom(x3), x3), 1)
        # ~ x = torch.cat((self.Rconv5(x), x4), 1)
        # x = torch.cat((self.Rconv4(x), x3), 1)
        x = torch.cat((self.Rconv3(x), x2), 1)
        x = torch.cat((self.Rconv2(x), x1), 1)
        y = self.Rconv1(x)
        return y


class U_Net3D_encoder(nn.Module):
    def __init__(self, fin, fout):
        super(U_Net3D_encoder, self).__init__()
        ndf = 32
        # ~ conNum = 64
        self.Lconv1 = nn.Sequential(
            nn.Conv3d(fin, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),
        )
        self.Lconv2 = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf, ndf * 2, 3, 1, 1),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm3d(ndf * 2, track_running_stats=False),
            nn.ReLU(True),
        )

        self.Lconv3 = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf*2, ndf*4, 3, 1, 1),
            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm3d(ndf*4,track_running_stats=False),
            nn.ReLU(True),

        )

        self.Lconv4 = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf*4, ndf*8, 3, 1, 1),
            nn.Conv3d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm3d(ndf*8,track_running_stats=False),
            nn.ReLU(True),
        )

        self.bottom_encoder = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf * 8, ndf * 16, 3, 1, 1),
            nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.BatchNorm3d(ndf * 16, track_running_stats=False),
            nn.ReLU(True),
            nn.AvgPool3d(2, 2),
        )

    def forward(self, x):
        x1 = self.Lconv1(x)
        x2 = self.Lconv2(x1)
        x3 = self.Lconv3(x2)
        x4 = self.Lconv4(x3)
        bottom = self.bottom_encoder(x4)
        # print(bottom.size())
        return bottom
        # return bottom

class U_Net3D_decoder(nn.Module):
    def __init__(self, fin, fout):
        super(U_Net3D_decoder, self).__init__()
        ndf = 32
        # ~ conNum = 64

        self.bottom_decoder = nn.Sequential(

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf * 16, ndf * 8, 1, 1, 0),
            nn.BatchNorm3d(ndf * 8, track_running_stats=False),
            nn.ReLU(True),
        )

        self.Rconv4 = nn.Sequential(
            nn.Conv3d(ndf * 16, ndf * 8, 3, 1, 1),
            nn.BatchNorm3d(ndf * 8, track_running_stats=False),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf * 8, ndf * 4, 1, 1, 0),
            nn.BatchNorm3d(ndf * 4, track_running_stats=False),
            nn.ReLU(True),

        )

        self.Rconv3 = nn.Sequential(
            nn.Conv3d(ndf * 8, ndf * 4, 3, 1, 1),
            nn.BatchNorm3d(ndf * 4, track_running_stats=False),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf * 4, ndf * 2, 1, 1, 0),
            nn.BatchNorm3d(ndf * 2, track_running_stats=False),
            nn.ReLU(True),
        )

        self.Rconv2 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 2, 3, 1, 1),
            nn.BatchNorm3d(ndf * 2, track_running_stats=False),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ndf * 2, ndf, 1, 1, 0),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),
        )

        self.Rconv1 = nn.Sequential(
            nn.Conv3d(ndf * 2, fout, 3, 1, 1),
            nn.BatchNorm3d(fout, track_running_stats=False),
            nn.ReLU(True),
        )

    def forward(self, Xs):
        x1, x2, x3, x4, bottom = Xs

        x = torch.cat((self.bottom_decoder(bottom), x4), 1)
        x = torch.cat((self.Rconv4(x), x3), 1)
        x = torch.cat((self.Rconv3(x), x2), 1)
        x = torch.cat((self.Rconv2(x), x1), 1)
        y = self.Rconv1(x)
        return y


class GraphCorrelationLayer_b(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """


    def __init__(self, in_features, out_features, concat=True):
        super(GraphCorrelationLayer_b, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.f_out = 256
        self.s_out = 3

        self.FC = nn.Parameter(torch.zeros(size=(in_features, self.out_features)))
        nn.init.xavier_uniform_(self.FC.data, gain=1.414)

        self.FW = nn.Parameter(torch.zeros(size=(in_features, self.f_out)))
        nn.init.xavier_uniform_(self.FW.data, gain=1.414)
        self.Fa = nn.Parameter(torch.zeros(size=(2 * (self.f_out), 1)))
        nn.init.xavier_uniform_(self.Fa.data, gain=1.414)

        self.SW = nn.Parameter(torch.zeros(size=(3, self.s_out)))
        nn.init.xavier_uniform_(self.SW.data, gain=1.414)
        self.Sa = nn.Parameter(torch.zeros(size=(2 * self.s_out, 1)))
        nn.init.xavier_uniform_(self.Sa.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.feature_correlation_distance = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0),
            # nn.ReLU(),
            # nn.Conv1d(16, 1, 1, 1, 0),
            # nn.Tanh(),
            nn.LeakyReLU(0.2),
            # nn.ELU(),
        )

        self.spatial_correlation_distance = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0),
            # nn.ReLU(),
            # nn.Conv1d(16, 1, 1, 1, 0),
            # nn.Tanh(),
            nn.LeakyReLU(0.2),
            # nn.ELU(),
        )

        self.correlation_similarity = nn.Sequential(
            nn.Linear(in_features + 3, 1),
            # nn.ELU(),
            nn.LeakyReLU(0.2),
            # nn.Sigmoid()
            # nn.Linear(256, 1),
        )

        # self.liner = nn.Linear(2, 1)

        self.relu = nn.ReLU()
        self.leak_relu = nn.LeakyReLU(0.2)

        self.elu = nn.ELU()

    def forward(self, features, coordinates, adj):
        N = features.size()[0]

        # print(coordinates.size())

        f_input = torch.mm(features, self.FC)
        f_mapping = torch.cat([f_input.repeat(1, N).view(N * N, -1).unsqueeze(1), f_input.repeat(N, 1).unsqueeze(1)], dim=1)
        f_mapping = self.feature_correlation_distance(f_mapping).squeeze(1)

        s_mapping = torch.cat([coordinates.repeat(1, N).view(N * N, -1).unsqueeze(1), coordinates.repeat(N, 1).unsqueeze(1)], dim=1)
        s_mapping = self.spatial_correlation_distance(s_mapping).squeeze(1)

        mapping = torch.cat([f_mapping, s_mapping], dim=1)

        feature_attention = self.correlation_similarity(mapping).view(N, N)
        # print(feature_attention.size())


        # f_mapping = self.relu(torch.mm(f_input, self.FW))

        # s_mapping = torch.mm(coordinates, self.SW)
        # mapping = torch.cat([coordinates, f_mapping], dim=1)

        # mapping = torch.cat([f_mapping.repeat(1, N).view(N * N, -1), f_mapping.repeat(N, 1)], dim=1).view(N, -1, 2 * (self.f_out))
        # feature_attention = torch.matmul(mapping, self.Fa).squeeze(2)

        # print(f_mapping.size())
        # feature_attention = self.relu(torch.matmul(mapping, self.Fa).squeeze(2))

        # feature_attention = torch.matmul(f_mapping, f_mapping.permute(1, 0))

        # print(f_mapping.size())
        # print(feature_attention.size())

        # f_mapping = torch.cat([f_mapping.repeat(1, N).view(N * N, -1), f_mapping.repeat(N, 1)], dim=1).view(N, -1, 2 * self.f_out)
        # print(f_mapping.size())
        # feature_attention = self.relu(torch.matmul(f_mapping, self.Fa).squeeze(2))

        # s_mapping = torch.mm(coordinates, self.SW)
        # spatial_attention = torch.matmul(s_mapping, s_mapping.permute(1, 0))
        # s_mapping = torch.cat([coordinates.repeat(1, N).view(N * N, -1), coordinates.repeat(N, 1)], dim=1).view(N, -1, 2 * self.s_out)
        # spatial_attention = self.relu(torch.matmul(s_mapping, self.Sa).squeeze(2))

        # attention = feature_attention * spatial_attention

        # zero_vec = -9e15 * torch.ones_like(feature_attention)
        # print (adj)
        # attention = torch.where(adj > 0, feature_attention, zero_vec)

        attention = feature_attention

        # attention = self.relu(torch.matmul(torch.cat([feature_attention.unsqueeze(-1), spatial_attention.unsqueeze(-1)], dim=2), self.a).squeeze(2))
        # print(attention.size())

        # attention_sum =

        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, f_input)
        # print(h_prime.size())
        # return h_prime
        if self.concat:
            # return self.leak_relu(h_prime)
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GraphCorrelationLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(GraphCorrelationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.f_out = 256
        self.s_out = 3

        self.FC = nn.Parameter(torch.zeros(size=(in_features, self.out_features)))
        nn.init.xavier_uniform_(self.FC.data, gain=1.414)

        self.FW = nn.Parameter(torch.zeros(size=(in_features, self.f_out)))
        nn.init.xavier_uniform_(self.FW.data, gain=1.414)
        self.Fa = nn.Parameter(torch.zeros(size=(2 * (self.f_out), 1)))
        nn.init.xavier_uniform_(self.Fa.data, gain=1.414)

        self.SW = nn.Parameter(torch.zeros(size=(3, self.s_out)))
        nn.init.xavier_uniform_(self.SW.data, gain=1.414)
        self.Sa = nn.Parameter(torch.zeros(size=(2 * self.s_out, 1)))
        nn.init.xavier_uniform_(self.Sa.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.feature_correlation_distance = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0),
            # nn.ReLU(),
            # nn.Conv1d(16, 1, 1, 1, 0),
            nn.Tanh(),
            # nn.LeakyReLU(0.2),
            # nn.ELU(),
        )

        self.spatial_correlation_distance = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0),
            # nn.ReLU(),
            # nn.Conv1d(16, 1, 1, 1, 0),
            nn.Tanh(),
            # nn.LeakyReLU(0.2),
            # nn.ELU(),
        )

        self.correlation_similarity = nn.Sequential(
            nn.Linear(in_features + 3, 1),
            # nn.ELU(),
            # nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Linear(256, 1),
        )

        self.self_attention = nn.Sequential(
            nn.Linear(515, 256),
            nn.Tanh(),
            nn.Linear(256, 17),
            nn.LeakyReLU(0.2),
        )

        # self.liner = nn.Linear(2, 1)

        self.relu = nn.ReLU()
        self.leak_relu = nn.LeakyReLU(0.2)

        self.elu = nn.ELU()

    def forward(self, features, coordinates, adj):
        N = features.size()[0]

        # print(coordinates.size())
        f_input = self.relu(torch.mm(features, self.FC))
        f_mapping = torch.cat([f_input.repeat(1, N).view(N * N, -1).unsqueeze(1), f_input.repeat(N, 1).unsqueeze(1)], dim=1)
        f_mapping = self.feature_correlation_distance(f_mapping).squeeze(1)

        s_mapping = torch.cat([coordinates.repeat(1, N).view(N * N, -1).unsqueeze(1), coordinates.repeat(N, 1).unsqueeze(1)], dim=1)
        s_mapping = self.spatial_correlation_distance(s_mapping).squeeze(1)

        mapping = torch.cat([f_mapping, s_mapping], dim=1)

        attention = self.correlation_similarity(mapping).view(N, N) + 1e-9

        # attention[attention < 0] =

        # zero_vec = torch.zeros_like(attention)
        # attention = torch.where(attention > 0, attention, zero_vec)

        # print(attention)
        # attention = F.softmax(attention, dim=1)
        attention_sum = torch.sum(attention, dim=1)
        # print(attention_sum.size())
        # attention = attention / attention_sum

        attention = (attention.permute(1, 0) / attention_sum).permute(1, 0)

        # print(torch.sum(attention,dim=1))

        print(attention)
        h_prime = torch.matmul(attention, f_input)
        s_prime = torch.matmul(attention, coordinates)
        # print(h_prime.size())
        # print(s_prime.size())
        # return h_prime
        if self.concat:
            # return self.leak_relu(h_prime)
            return h_prime, s_prime, attention
        else:
            return h_prime, s_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphCorrelationLayer_pre(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(GraphCorrelationLayer_pre, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.f_out = 256
        self.s_out = 3


        self.FC = nn.Parameter(torch.zeros(size=(in_features, self.out_features)))
        nn.init.xavier_uniform_(self.FC.data, gain=1.414)

        self.attention = nn.Parameter(torch.zeros(17, 17) - 1 + torch.eye(17, 17) * 2)


        # self.liner = nn.Linear(2, 1)

        self.relu = nn.ReLU()
        self.leak_relu = nn.LeakyReLU(0.2)

        self.elu = nn.ELU()

    def forward(self, features, coordinates, adj):
        N = features.size()[0]

        # print(attention)
        f_input = self.relu(torch.mm(features, self.FC))
        attention = F.softmax(self.attention, dim=1)
        # attention_sum = torch.sum(self.attention, dim=1)
        # print(attention_sum.size())

        # attention = (self.attention.permute(1, 0) / attention_sum).permute(1, 0)

        # print(torch.sum(attention,dim=1))


        # print(attention)
        h_prime = torch.matmul(attention, f_input)
        # s_prime = torch.matmul(attention, coordinates)
        # print(h_prime.size())
        # print(s_prime.size())
        # return h_prime
        if self.concat:
            # return self.leak_relu(h_prime)
            return h_prime
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphCorrelationLayer_selfattention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(GraphCorrelationLayer_selfattention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.f_out = 256
        self.s_out = 3


        self.FC = nn.Parameter(torch.zeros(size=(in_features, self.out_features)))
        nn.init.xavier_uniform_(self.FC.data, gain=1.414)

        self.self_attention = nn.Sequential(
            nn.Linear(515, 256),
            nn.Tanh(),
            nn.Linear(256, 17),
            # nn.LeakyReLU(0.2),
        )


        # self.liner = nn.Linear(2, 1)

        self.relu = nn.ReLU()
        self.leak_relu = nn.LeakyReLU(0.2)

        self.elu = nn.ELU()

    def forward(self, features, coordinates, adj):
        N = features.size()[0]

        # print(attention)
        f_input = self.relu(torch.mm(features, self.FC))
        f_input_cat = torch.cat([f_input, coordinates], dim=1)
        attention = self.self_attention(f_input_cat).permute(1, 0)
        attention = F.softmax(attention, dim=1)
        # attention_sum = torch.sum(self.attention, dim=1)
        # print(attention_sum.size())

        # attention = (self.attention.permute(1, 0) / attention_sum).permute(1, 0)

        # print(torch.sum(attention,dim=1))


        # print(attention)
        h_prime = torch.matmul(attention, f_input)
        # s_prime = torch.matmul(attention, coordinates)
        # print(h_prime.size())
        # print(s_prime.size())
        # return h_prime
        if self.concat:
            # return self.leak_relu(h_prime)
            return h_prime
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class structure_attention_net(nn.Module):
    def __init__(self, landmarkNum):
        super(structure_attention_net, self).__init__()
        self.landmarkNum = landmarkNum
        # self.out_att = GraphCorrelationLayer_pre(512, 512, concat=True)
        # self.out_atts = [GraphCorrelationLayer_pre(512, 512, concat=True) for _ in range(8)]
        self.out_atts = [GraphCorrelationLayer_selfattention(512, 512, concat=True) for _ in range(8)]
        for i, out_att in enumerate(self.out_atts):
            self.add_module('out_att_{}'.format(i), out_att)
        self.fc = nn.Sequential(
            nn.Conv1d(8, 1, 1, 1, 0),
            # nn.Linear(512, 256),
            nn.ReLU(),
        )

    def forward(self, Xs, ROIs_b, adj):
        Xs = torch.cat([X.view(1, -1) for X in Xs], dim=0)
        # print(Xs.size())
        # Xs = torch.cat([self.out_att(Xs, ROIs_b, adj), self.fc(Xs)], dim=1)
        # print(Xs.size())
        # Xs = self.out_att(Xs, ROIs_b)
        # Xs = self.fc(torch.cat([self.out_atts[i](Xs, ROIs_b, adj).unsqueeze(1) for i in range(8)], dim=1)).squeeze()
        Xs = self.fc(torch.cat([self.out_atts[i](Xs, ROIs_b, adj).unsqueeze(1) for i in range(8)], dim=1)).squeeze() + Xs
        # print(Xs.size())
        # Xs = torch.cat([self.fc(torch.cat([self.out_atts[i](Xs, ROIs_b) for i in range(8)], dim=1)), Xs], dim=1)
        # Xs = self.fc(torch.cat([self.out_atts[i](Xs, ROIs_b) for i in range(8)], dim=1)) * Xs
        # print(Xs.size())

        # Xs, ROIs_t, attention = self.out_att(Xs, ROIs_b, adj)

        ROIs_t = ROIs_b
        attention = 0
        # Xs = self.fc(torch.cat([self.out_att(Xs, ROIs_b, adj).unsqueeze(1), Xs.unsqueeze(1)], dim=1)).squeeze()
        # print(Xs.size())
        Xs = [Xs[i].view(1, -1, 1, 1, 1) for i in range(self.landmarkNum)]
        return Xs, ROIs_t, attention

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.relu = nn.ReLU()

    def forward(self, input):
        h = torch.mm(input, self.W)

        # h = torch.Tensor([[1, 11], [2, 22], [3, 33]]).cuda(0)
        N = h.size()[0]
        # print (h.size())

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        # print (h.repeat(1, N))
        # print (h.repeat(1, N).view(N * N, -1))
        # print (h.repeat(N, 1))
        # print (torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1))
        # print (a_input.size())
        e = self.relu(torch.matmul(a_input, self.a).squeeze(2))
        # print (torch.matmul(a_input, self.a).size())
        # print (e.size())
        # print (e.size())
        zero_vec = -9e15 * torch.ones_like(e)
        # print (adj)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = e
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        # print(h_prime.size())

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class decoder_offset(nn.Module):
    def __init__(self, fin, fout):
        super(decoder_offset, self).__init__()
        self.liner = nn.Sequential(
            # nn.AvgPool3d(2, 2),
            nn.Conv3d(fin, fout, 1, 1, 0),
        )

    def forward(self, x):
        return self.liner(x)

class decoder_heatmap(nn.Module):
    def __init__(self, fin, fout):
        super(decoder_heatmap, self).__init__()
        self.liner = nn.Sequential(
            nn.Conv3d(fin, fout, 1, 1, 0),
        )

    def forward(self, x):
        return self.liner(x)

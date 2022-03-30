from __future__ import print_function, division
import torch
import torch.nn as nn

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

    
class embedding_net(nn.Module):
    def __init__(self, fin, fout):
        super(embedding_net, self).__init__()
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
        embedding = self.bottom_encoder(x4)
        # print(bottom.size())
        return embedding
        # return bottom

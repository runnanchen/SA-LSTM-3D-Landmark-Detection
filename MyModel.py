from __future__ import print_function, division
import torch
import torch.nn as nn
import MyUtils
import MyNetworkLayer as MNL

class coarseNet(nn.Module):
    def __init__(self, landmarksNum, useGPU, image_scale):
        super(coarseNet, self).__init__()
        self.landmarkNum = landmarksNum
        self.usegpu = useGPU
        self.image_scale = image_scale
        self.u_net = MNL.U_Net3D(1, 64)
        self.conv3d = nn.Sequential(
            nn.Conv3d(64, landmarksNum, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        global_features = self.u_net(x)
        x = self.conv3d(global_features)
        heatmap_sum = torch.sum(x.view(self.landmarkNum, -1), dim=1)
        global_heatmap = [x[0, i, :, :, :].squeeze() / heatmap_sum[i] for i in range(self.landmarkNum)]

        return global_heatmap, global_features

class fine_LSTM(nn.Module):
    def __init__(self, landmarkNum, usegpu, iteration, crop_size):
        super(fine_LSTM, self).__init__()
        self.landmarkNum = landmarkNum
        self.usegpu = usegpu
        self.encoder = MNL.U_Net3D_encoder(1, 64)
        self.iteration = iteration
        self.crop_size = crop_size
        self.size_tensor = torch.tensor([1 / 767, 1 / 767, 1 / 575]).cuda(self.usegpu)
        self.decoders_offset_x = nn.Conv1d(landmarkNum, landmarkNum, 512 + 64, 1, 0, groups=landmarkNum)
        self.decoders_offset_y = nn.Conv1d(landmarkNum, landmarkNum, 512 + 64, 1, 0, groups=landmarkNum)
        self.decoders_offset_z = nn.Conv1d(landmarkNum, landmarkNum, 512 + 64, 1, 0, groups=landmarkNum)
        self.attention_gate_share = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.Tanh(),
            # nn.Linear(256, 1)
            # nn.Conv1d(landmarkNum, landmarkNum, 256, 1, 0, groups=landmarkNum),
        )
        self.attention_gate_head = nn.Conv1d(landmarkNum, landmarkNum, 256, 1, 0, groups=landmarkNum)
        self.graph_attention = MNL.graph_attention(64, usegpu)

    def forward(self, coarse_landmarks, labels, inputs_origin, coarse_feature, phase, size_tensor_inv):
        h_state = 0
        predicts = []
        c_state = 0
        predict = coarse_landmarks.detach()

        for i in range(0, self.iteration):
            ROIs = 0
            if phase == 'train':
                if i == 0:
                    # ROIs = labels + torch.from_numpy(np.random.uniform(-32.0 / 768, 32.0 / 768, labels.size())).cuda(self.usegpu).float()
                    ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=32.0 / 768 / 3, size = labels.size())).cuda(self.usegpu).float()
                elif i == 1:
                    # ROIs = labels + torch.from_numpy(np.random.uniform(-24.0 / 768, 24.0 / 768, labels.size())).cuda(self.usegpu).float()
                    ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=16.0 / 768 / 3, size = labels.size())).cuda(self.usegpu).float()
                else:
                    # ROIs = labels + torch.from_numpy(np.random.uniform(-16.0 / 768, 16.0 / 768, labels.size())).cuda(self.usegpu).float()
                    ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=8.0 / 768 / 3, size = labels.size())).cuda(self.usegpu).float()
            else:
                ROIs = predict

            ROIs = MyUtils.adjustment(ROIs, labels)
            cropedtems = MyUtils.getcropedInputs_related(ROIs.detach().cpu().numpy(), labels, inputs_origin, -1, i)
            cropedtems = torch.cat([cropedtems[i].cuda(self.usegpu) for i in range(len(cropedtems))], dim=0)
            features = self.encoder(cropedtems).squeeze().unsqueeze(0)
            
            global_feature = MyUtils.get_global_feature(ROIs.detach().cpu().numpy(), coarse_feature, self.landmarkNum)
            global_feature = self.graph_attention(ROIs, global_feature)
            features = torch.cat((features, global_feature), dim=2)

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

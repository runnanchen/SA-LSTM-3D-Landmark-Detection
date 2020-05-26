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
        self.embedding = MNL.embedding_net(1, 64)
        self.iteration = iteration
        self.crop_size = crop_size

        self.size_tensor = torch.tensor([1 / 767, 1 / 767, 1 / 575])

        self.decoders_offset_x = nn.Conv1d(landmarkNum, landmarkNum, 512, 1, 0, groups=landmarkNum)
        self.decoders_offset_y = nn.Conv1d(landmarkNum, landmarkNum, 512, 1, 0, groups=landmarkNum)
        self.decoders_offset_z = nn.Conv1d(landmarkNum, landmarkNum, 512, 1, 0, groups=landmarkNum)

        self.attention_gate_share = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
        )
        self.attention_gate_head = nn.Conv1d(landmarkNum, landmarkNum, 256, 1, 0, groups=landmarkNum)


    def forward(self, ROIs, labels, inputs_origin):

        cropedtems = MyUtils.getcropedInputs(ROIs.detach().cpu().numpy(), inputs_origin, self.crop_size[0], -1)
        cropedtems = torch.cat([cropedtems[i] for i in range(len(cropedtems))], dim=0)
        embeddings = self.embedding(cropedtems).squeeze().unsqueeze(0)

        x, y, z = self.decoders_offset_x(embeddings), self.decoders_offset_y(embeddings), self.decoders_offset_z(embeddings)
        predict = torch.cat([x, y, z], dim=2) * self.size_tensor + torch.from_numpy(ROIs)

        h_state = embeddings
        predicts = [predict]
        c_state = torch.from_numpy(ROIs)

        for i in range(self.iteration - 1):

            ROIs = predict

            cropedtems = MyUtils.getcropedInputs(ROIs.detach().cpu().numpy(), inputs_origin, self.crop_size[0], -1)
            cropedtems = torch.cat([cropedtems[i].cuda(self.usegpu) for i in range(len(cropedtems))], dim=0)
            embeddings = self.embedding(cropedtems).squeeze().unsqueeze(0)

            gate_f = self.attention_gate_head(self.attention_gate_share(h_state.squeeze()).unsqueeze(0))
            gate_a = self.attention_gate_head(self.attention_gate_share(embeddings.squeeze()).unsqueeze(0))
            gate = torch.softmax(torch.cat([gate_f, gate_a], dim=2), dim=2)

            h_state = h_state * gate[0, :, 0].view(1, -1, 1) + embeddings * gate[0, :, 1].view(1, -1, 1)
            c_state = c_state * gate[0, :, 0].view(1, -1, 1) + ROIs * gate[0, :, 1].view(1, -1, 1)

            x, y, z = self.decoders_offset_x(h_state), self.decoders_offset_y(h_state), self.decoders_offset_z(h_state)
            predict = torch.cat([x, y, z], dim=2) * self.size_tensor.cuda(self.usegpu) + c_state.detach()
            predicts.append(predict)

        predicts = torch.cat(predicts, dim=0)

        return predicts
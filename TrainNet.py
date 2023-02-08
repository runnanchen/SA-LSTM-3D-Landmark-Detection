from __future__ import print_function, division
import torch
import numpy as np
import time
import MyUtils
import torch.nn.functional as F
import processData
import LossFunction
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def train_model(coarse_net, fine_LSTM, dataloaders, criterion_coarse, criterion_fine, optimizer, config
                ):
    since = time.time()
    test_epoch = 5

    gl, gh, gw = config.image_scale
    global_coordinate = torch.ones(gl, gh, gw, 3).float()
    for i in range(gl):
        global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
    for i in range(gh):
        global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
    for i in range(gw):
        global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
    global_coordinate = global_coordinate.cuda(config.use_gpu) * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)]).cuda(config.use_gpu)

    cl, ch, cw = config.crop_size
    base_coordinate = torch.ones(cl, ch, cw, 3).float().cuda(config.use_gpu)
    for i in range(cl):
        base_coordinate[i, :, :, 0] = base_coordinate[i, :, :, 0] * i
        base_coordinate[:, i, :, 1] = base_coordinate[:, i, :, 1] * i
        base_coordinate[:, :, i, 2] = base_coordinate[:, :, i, 2] * i
    base_coordinate = (base_coordinate - cl/2)

    for epoch in range(config.epochs):

        train_fine_Off = []
        train_fine_Off_heatmap = []
        train_coarse_Off = []
        test_fine_Off = []
        test_fine_Off_heatmap = []
        test_coarse_Off = []


        for phase in ['train', 'val']:
            # print ("1")
            # datas = DataLoader(dataloaders[phase], batch_size=1, shuffle=True, num_workers=0)
            datas = DataLoader(dataloaders[phase], batch_size=1, shuffle=False, num_workers=0)

            pbar = tqdm(total=len(datas))

            if phase == 'train':
                if config.stage == 'test': continue
                coarse_net.train(True)  # Set model to training mode
                fine_LSTM.train(True)
            else:
                # ~ continue
                if epoch % test_epoch != 0:
                    continue
                coarse_net.train(False)  # Set model to evaluate mode
                fine_LSTM.train(False)

            # Iterate over data.
            lent = len(datas)
            running_loss = 0

            # for ide in range(lent):
            t = 0
            for data in datas:
                t += 1
                inputs, inputs_origin, labels, image_name, size, DICOM_origin_vis = data['DICOM'].cuda(config.use_gpu), data['DICOM_origin'], data[
                    'landmarks'].cuda(config.use_gpu).squeeze(0), data['imageName'], data['size'], data['DICOM_origin_vis']
                size_tensor = torch.tensor([size[1], size[2], size[0]]).cuda(config.use_gpu)
                size_tensor_inv = 1.0 / size_tensor.float()
                # print(labels.size())
                optimizer.zero_grad()

                coarse_heatmap, coarse_feature = coarse_net(inputs)
                coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate).unsqueeze(0)
                fine_landmarks = fine_LSTM(coarse_landmarks, labels, inputs_origin, coarse_feature, phase, size_tensor_inv)

                loss = torch.abs(fine_landmarks - labels).sum()
                loss += criterion_coarse(coarse_heatmap, global_coordinate, labels, phase)

                # backward + optimize only if in training phase
                if phase == 'train' and config.stage == 'train':
                    loss.backward()
                    optimizer.step()

                if epoch % test_epoch == 0:
                    coarse_off = MyUtils.get_coarse_errors(coarse_landmarks, global_coordinate, labels, size_tensor)
                    fine_off = MyUtils.get_fine_errors(fine_landmarks, labels, size_tensor)

                    fine_off_heatmap = coarse_off
                    if phase == "train":
                        train_fine_Off.append(fine_off.detach().cpu())
                        train_fine_Off_heatmap.append(fine_off_heatmap.detach().cpu())
                        train_coarse_Off.append(coarse_off.detach().cpu())
                    else:
                        test_fine_Off.append(fine_off.detach().cpu())
                        test_fine_Off_heatmap.append(fine_off_heatmap.detach().cpu())
                        test_coarse_Off.append(coarse_off.detach().cpu())
                # statistics
                running_loss += loss.item()
                pbar.update(1)

            # print (dataset_sizes[phase])
            epoch_loss = running_loss / lent
            pbar.close()
            epoch_acc = 0
            if epoch % 1 == 0:
                print('{} epoch: {} Loss: {}'.format(
                    phase, epoch, epoch_loss))
        if epoch % test_epoch == 0:


            test_fine_Off = torch.cat(test_fine_Off, dim=0)
            test_fine_Off_heatmap = torch.cat(test_fine_Off_heatmap, dim=0)
            test_coarse_Off = torch.cat(test_coarse_Off, dim=0)

            if config.stage == 'train':
                train_fine_Off = torch.cat(train_fine_Off, dim=0)
                train_fine_Off_heatmap = torch.cat(train_fine_Off_heatmap, dim=0)
                train_coarse_Off = torch.cat(train_coarse_Off, dim=0)
            else:
                train_fine_Off = test_fine_Off
                train_fine_Off_heatmap = test_fine_Off_heatmap
                train_coarse_Off = test_coarse_Off

                # torch.save(test_fine_Off, 'MICCAI_fine.pt')

            test_coarse_SDR, test_coarse_SD, test_coarse_MRE = MyUtils.analysis_result(config.landmarkNum, test_coarse_Off.numpy())
            train_coarse_SDR, train_coarse_SD, train_coarse_MRE = MyUtils.analysis_result(config.landmarkNum, train_coarse_Off.numpy())

            test_fine_SDR, test_fine_SD, test_fine_MRE = MyUtils.analysis_result(config.landmarkNum, test_fine_Off.numpy())
            train_fine_SDR, train_fine_SD, train_fine_MRE = MyUtils.analysis_result(config.landmarkNum, train_fine_Off.numpy())

            test_fine_heatmap_SDR, test_fine_heatmap_SD, test_fine_heatmap_MRE = MyUtils.analysis_result(config.landmarkNum, test_fine_Off_heatmap.numpy())
            train_fine_heatmap_SDR, train_fine_heatmap_SD, train_fine_heatmap_MRE = MyUtils.analysis_result(config.landmarkNum, train_fine_Off_heatmap.numpy())

            for landmarkId in range(config.landmarkNum):
                print(landmarkId, ": ", processData.landmarkList[landmarkId], train_coarse_MRE[landmarkId], " ",
                      train_coarse_SD[landmarkId], " ", train_coarse_SDR[landmarkId])
            print()

            for landmarkId in range(config.landmarkNum):
                print(landmarkId, ": ", processData.landmarkList[landmarkId], test_coarse_MRE[landmarkId], " ",
                      test_coarse_SD[landmarkId], " ", test_coarse_SDR[landmarkId])

            print()
            for landmarkId in range(config.landmarkNum):
                print(landmarkId, ": ", processData.landmarkList[landmarkId], train_fine_MRE[landmarkId], " ",
                      train_fine_SD[landmarkId], " ", train_fine_SDR[landmarkId])
            print()

            for landmarkId in range(config.landmarkNum):
                print(landmarkId, ": ", processData.landmarkList[landmarkId], test_fine_MRE[landmarkId], " ",
                      test_fine_SD[landmarkId], " ", test_fine_SDR[landmarkId])

            print("train_coarse_avgOff(train_SD) ", np.mean(train_coarse_MRE), "+-", np.mean(train_coarse_SD), np.mean((train_coarse_SDR), 0))
            print("test_coarse_avgOff(test_SD) ", np.mean(test_coarse_MRE), "+-", np.mean(test_coarse_SD), np.mean((test_coarse_SDR), 0))

            print("train_fine_avgOff(train_SD) ", np.mean(train_fine_MRE), "+-", np.mean(train_fine_SD), np.mean((train_fine_SDR), 0))
            print("test_fine_avgOff(test_SD) ", np.mean(test_fine_MRE), "+-", np.mean(test_fine_SD), np.mean((test_fine_SDR), 0))

            print("train_fine_heatmap_avgOff(train_SD) ", np.mean(train_fine_heatmap_MRE), "+-", np.mean(train_fine_heatmap_SD), np.mean((train_fine_heatmap_SDR), 0))
            print("test_fine_heatmap_avgOff(test_SD) ", np.mean(test_fine_heatmap_MRE), "+-", np.mean(test_fine_heatmap_SD), np.mean((test_fine_heatmap_SDR), 0))

            # torch.save(coarse_net, "output/" + str(epoch) + saveName + 'coarse.pkl')
            # torch.save(fine_LSTM, "output/" + str(epoch) + saveName + 'fine_LSTM.pkl')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def test_model(corseNet, fineNet, dataloaders, criterion1, criterion2, optimizer, config):
    since = time.time()

    # num_epochs, use_gpu, R1, R2, saveName, landmarkNum, image_scale

    best_acc = [0, 0, 0, 0, 0, 0]
    test_avgOff = 0
    # Each epoch has a training and validation phase
    phase = 'val'
    corseNet.train(False)  # Set model to evaluate mode
    fineNet.train(False)

    # Iterate over data.
    lent = len(dataloaders[phase])
    test_Off = np.zeros((0, config.landmarkNum))
    for ide in range(lent):
        data = dataloaders[phase][ide]
        inputs, inputs_origin, labels = data['DICOM'].cuda(config.use_gpu), data['DICOM_origin'], data['landmarks']
        optimizer.zero_grad()

        heatMapsCorse, coordinatesCorse = corseNet(inputs)

        coordinatesCorse = coordinatesCorse.unsqueeze(0)

        ROIs = coordinatesCorse.cpu().detach().numpy()

        cropedtem = MyUtils.getcropedInputs(ROIs, inputs_origin, 64, -1)
        cropedInputs = [cropedInput.cuda(config.use_gpu) for cropedInput in cropedtem]
        data['cropedInputs'] = cropedInputs

        cropedInputs = data['cropedInputs']
        outputs2 = 0
        heatMapsFine, coordinatesFine = fineNet(ROIs, cropedInputs, outputs2)
        coordinatesFine = coordinatesFine.unsqueeze(0)
        print(coordinatesFine)
        coorall = coordinatesCorse + coordinatesFine
        # coorall = coordinatesCorse
        off = MyUtils.getCoordinate_new(heatMapsCorse, heatMapsFine, labels, config.R1, config.R2, config.use_gpu, coordinatesCorse,
                                        coordinatesFine, config)  # getCoordinate_new1

        print(off)

    time_elapsed = time.time() - since
    print('test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

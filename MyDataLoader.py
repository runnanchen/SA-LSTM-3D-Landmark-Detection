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


class ZipDataset(Dataset):
    def __init__(self, root_path, cache_into_memory=False):
        if cache_into_memory:
            f = open(root_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(root_path, 'r')
        self.name_list = list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist()))
        self.to_tensor = ToTensor()

    def __getitem__(self, key):
        buf = self.zip_file.read(name=self.name_list[key])
        img = self.to_tensor(cv2.imdecode(np.fromstring(buf, dtype=np.uint8), cv2.IMREAD_COLOR))
        return img

    def __len__(self):
        return len(self.name_list)


'''
if __name__ == '__main__':
    dataset = ZipDataset('COCO.zip', cache_into_memory=False)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for batch_idx, sample in enumerate(dataloader):
        print(batch_idx, sample.size())
'''


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, input_size):
        # assert isinstance(input_size, (int, tuple))
        self.input_size = input_size

    def __call__(self, sample):
        # image, landmarks, targetMaps = sample['image'], sample['landmarks'], sample['targetMaps']
        DICOM, DICOM_origin, landmarks, imageName = sample['DICOM'], sample['DICOM_origin'], sample['landmarks'], \
                                                    sample['imageName']
        # l, h, w = 576, 768, 768
        l, h, w = self.input_size

        # ~ l, h, w = DICOM.shape[:3]
        # ~ newl, newh, neww = self.output_size
        # ~ scalel, scaleh, scalew = newl/l, newh/h, neww/w

        # ~ newDICOM = zoom(DICOM, (scalel, scaleh, scalew))
        print(DICOM.shape)
        landmarks = landmarks * [1 / (h - 1), 1 / (w - 1), 1 / (l - 1)]

        return {'DICOM': DICOM, 'DICOM_origin': DICOM_origin, 'landmarks': landmarks, 'imageName': imageName}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, tlandmarks, targetMaps = sample['image'], sample['landmarks'], sample['targetMaps']

        DICOM, DICOM_origin, landmarks, imageName = sample['DICOM'], sample['DICOM_origin'], sample['landmarks'], \
                                                    sample['imageName']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # ~ image = image.transpose((2, 0, 1))
        # ~ DICOM  = (DICOM-np.min(DICOM))/(np.max(DICOM) - np.min(DICOM))
        DICOM = (DICOM - np.mean(DICOM)) / np.std(DICOM)

        # import pdb
        # pdb.set_trace()

        # crop_list = []
        size = DICOM_origin.shape
        DICOM_origin = (DICOM_origin - np.mean(DICOM_origin)) / np.std(DICOM_origin)
        DICOM_origin = torch.from_numpy(DICOM_origin).float().unsqueeze(0).unsqueeze(0)
        crop_list = MyUtils.getcropedInputs(landmarks.reshape(1, 17, 3), DICOM_origin, 96, -1)
        crop_list = [item.squeeze(0).squeeze(0) for item in crop_list]


        return {'DICOM': torch.from_numpy(DICOM).float(),
                'DICOM_origin': crop_list,
                # 'DICOM_origin_vis': DICOM_origin,
                'DICOM_origin_vis': DICOM,
                'landmarks': torch.from_numpy(landmarks).float(),
                'size': size,
                'imageName': imageName}
        #
        # return {'DICOM': torch.from_numpy(DICOM).float().unsqueeze(0),
        #         'DICOM_origin': torch.from_numpy(DICOM_origin).float().unsqueeze(0),
        #         'landmarks': torch.from_numpy(landmarks).float(),
        #         'imageName': imageName}

class LandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, landmarksNum=17):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.landmarkNum = landmarksNum

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "72_" + self.landmarks_frame.iloc[idx, 0])
        img_name_origin = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = np.load(img_name)
        # image_origin = image
        image_origin = np.load(img_name_origin)
        # image = MyUtils.resizeDICOM(image_origin, (96, 128, 128))

        landmarks = self.landmarks_frame.iloc[idx, 1:self.landmarkNum * 3 + 1].values.astype('float')
        # print(landmarks.shape)
        landmarks = landmarks.reshape(-1, 3)

        sample = {'DICOM': image, 'DICOM_origin': image_origin, 'landmarks': landmarks, 'imageName': img_name_origin}
        if self.transform:
            sample = self.transform(sample)

        return sample

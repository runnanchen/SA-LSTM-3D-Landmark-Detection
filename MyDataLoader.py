from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        DICOM, DICOM_origin, landmarks, imageName = sample['DICOM'], sample['DICOM_origin'], sample['landmarks'], \
                                                    sample['imageName']
        l, h, w = 576, 768, 768
        landmarks = landmarks * [1 / (h - 1), 1 / (w - 1), 1 / (l - 1)]

        return {'DICOM': DICOM, 'DICOM_origin': DICOM_origin, 'landmarks': landmarks, 'imageName': imageName}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        DICOM, DICOM_origin, landmarks, imageName = sample['DICOM'], sample['DICOM_origin'], sample['landmarks'], \
                                                    sample['imageName']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        DICOM = (DICOM - np.mean(DICOM)) / np.std(DICOM)
        DICOM_origin = (DICOM_origin - np.mean(DICOM_origin)) / np.std(DICOM_origin)

        return {'DICOM': torch.from_numpy(DICOM).float().unsqueeze(0),
                'DICOM_origin': torch.from_numpy(DICOM_origin).float().unsqueeze(0),
                'landmarks': torch.from_numpy(landmarks).float(),
                'imageName': imageName}

class LandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, landmarksNum=19):
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
        image_origin = np.load(img_name_origin)

        landmarks = self.landmarks_frame.iloc[idx, 1:self.landmarkNum * 3 + 1].values.astype('float')
        landmarks = landmarks.reshape(-1, 3)

        sample = {'DICOM': image, 'DICOM_origin': image_origin, 'landmarks': landmarks, 'imageName': img_name_origin}
        if self.transform:
            sample = self.transform(sample)

        return sample

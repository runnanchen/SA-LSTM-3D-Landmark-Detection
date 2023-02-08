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
import json
import h5py
import pickle as pkl
import csv
import argparse
import sys
# import functional as F
from PIL import Image, ImageDraw, ImageFont
import MyUtils
import scipy.io as scio
# import dcmstack, dicom
from glob import glob
import pydicom
import matplotlib.pyplot as plt
from xml.dom import minidom
from xml.etree import ElementTree as ET

parser = argparse.ArgumentParser(description='preprocessing')
parser.add_argument('--root_dir', dest='root_dir',
                    help='root_dir',
                    default=".", type=str)

if len(sys.argv) == 1:
    parser.print_help()
# sys.exit(1)

'''
landmarkList = ["Or-R", "Or-L", "Fz-R", "Fz-L", "Zy-R", "Zy-L", "Go-R", "Go-L", "PMP-R", "PMP-L",
				"Po-R", "Po-L",	"Co-R", "Co-L", "N", "N'", "ANS", "PNS", "A", "Prn", "Sn", "Ls", "Sto", "Li",  
				"Ba", "B", "Sl", "Me", "Pog", "Gn", "Pog'", "Me'", "Gn'"]

landmarkList_low = ["or-r", "or-l", "fz-r", "fz-l", "zy-r", "zy-l", "go-r", "go-l", "pmp-r", "pmp-l",
				"po-r", "po-l",	"co-r", "co-l", "n", "n'", "ans", "pns", "a", "prn", "sn", "ls", "sto", "li",  
				"ba", "b", "sl", "me", "pog", "gn", "pog'", "me'", "gn'"]
'''


landmarkList = ["Or-R", "Or-L", "Zy-R", "Zy-L", "Go-R", "Go-L", "Po-R", "Po-L", "N", "ANS", "PNS", "A", "Ba", "B", "Me",
                "Pog", "Gn"]

landmarkList_low = ["or-r", "or-l", "zy-r", "zy-l", "go-r", "go-l", "po-r", "po-l", "n", "ans", "pns", "a", "ba", "b",
                    "me", "pog", "gn"]

args = parser.parse_args()

root_dir = args.root_dir
target_dir = "processed_data"
txt_dir = os.path.join(root_dir, "landmarks")
img_dir = os.path.join(root_dir, "DICOMs")
csv_file = os.path.join(target_dir, "test.csv")

def landmarkDataset(txt_dir, img_dir):
    tt = 0
    tx = 0
    ty = 0
    # panoDataSet = []
    # decayDataSet = []
    # perDataSet = []
    dataSet = []
    tt = 0
    t0 = 0
    out = open(csv_file, 'w')
    csv_writer = csv.writer(out)
    tt = 0
    fiLine = []
    DICOMList = ['15', '06', '14', '45', '35', '79', '75', '77', '68', '17', '78', '30', '40', '37', '27', '50', '91',
                 '94', '12', '01']
    fiLine.append("st".encode())
    # fiLine.append(str.encode("st"))
    for i in range(len(landmarkList)):
        wx = landmarkList[i] + "_x"
        wy = landmarkList[i] + "_y"
        wz = landmarkList[i] + "_z"
        wx, wy, wz = wx.encode('utf-8'), wy.encode('utf-8'), wz.encode('utf-8')
        # wx, wy, wz = str.encode(wx), str.encode(wy), str.encode(wz)


        fiLine.append(wx)
        fiLine.append(wy)
        fiLine.append(wz)
    # ~ fiLine.append(landmarkList[i])
    # ~ fiLine.append(landmarkList[i])
    # ~ fiLine.append(landmarkList[i])
    csv_writer.writerow(fiLine)
    for root, dirs, files in os.walk(txt_dir):
        for imageName in files:
            # ~ if imageName != "04-New": continue
            # imageName = imageName.split("-")[0]
            annoPath = os.path.join(txt_dir, imageName)
            if not os.path.exists(annoPath):
                print("error")
                continue

            # print (tt)

            imageDICOM = imageName.split("-")[0]
            print(imageDICOM)
            # if imageDICOM not in DICOMList: continue
            # ~ if imageDICOM != "52": continue

            imagePath = os.path.join(img_dir, imageDICOM)
            if not os.path.exists(imagePath):
                print("error")
                continue

            tt = tt + 1
            print(tt)
            src_paths = glob(os.path.join(imagePath, "*.dcm"))
            src_paths.sort(reverse=True)
            my_stack = []
            for src_path in src_paths:
                src_dcm = pydicom.read_file(src_path)
                src_dcm = src_dcm.pixel_array
                src_dcm = np.flipud(src_dcm)
                my_stack.append(src_dcm)


            # ~ # z, x, y
            DICOM = np.array(my_stack)

            DICOM_72 = MyUtils.resizeDICOM(DICOM, (72, 96, 96))
            # DICOM_96 = MyUtils.resizeDICOM(DICOM, (96, 128, 128))

            np.save(os.path.join(target_dir, "images", imageDICOM), DICOM)
            # np.savez_compressed(os.path.join(target_dir, "images", imageDICOM), DICOM)
            np.save(os.path.join(target_dir, "images", "72_"+imageDICOM), DICOM_72)

            print(annoPath)
            tree111 = ET.parse(annoPath)
            print("ok")
            root111 = tree111.getroot()
            # ~ print (root111)

            # ~ print (len(root111))
            landmarkCoor = np.zeros((len(landmarkList), 3))

            cland = []

            for fiducial in root111:
                landmarksName = fiducial.attrib['label'].split(" ")[1]
                # ~ print (landmarksName)


                landmarksName_tem = landmarksName.lower()

                if landmarksName_tem not in landmarkList_low:
                    continue

                x, y, z = fiducial.attrib['posx'], fiducial.attrib['posy'], fiducial.attrib['posz']
                x, y, z = int(round(float(x) / 0.3)), int(round(float(y) / 0.3)), int(round(float(z) / 0.3))
                x, y = y, x

                # ~ print (x, y, z)
                if x >= 768 or y >= 768 or z >= 576: continue
                if x < 0 or y < 0 or z < 0: continue

                cland.append(landmarksName_tem)

                landmarkId = landmarkList_low.index(landmarksName_tem)
                # ~ print (landmarkId, x, y, z)
                # ~ DICOM = MyUtils.resizeDICOM(DICOM, (384, 700, 700))
                # x, y, z = int(round(x/6)), int(round(y/6)), int(round(z/6))

                landmarkCoor[landmarkId, :] = np.array([x, y, z])
            # ~ MyUtils.showDICOM(DICOM, x, y, z)

            line = []
            line.append(imageDICOM + ".npy")

            skip = False
            for item in landmarkList_low:
                if item not in cland:
                    # ~ line.append(item)
                    skip = True
            if skip: continue

            lengb = landmarkCoor.shape[0]
            for i in range(lengb):
                landmarkX = landmarkCoor[i][0]
                landmarkY = landmarkCoor[i][1]
                landmarkZ = landmarkCoor[i][2]

                line.append(str(landmarkX).encode())
                line.append(str(landmarkY).encode())
                line.append(str(landmarkZ).encode())

            csv_writer.writerow(line)
    print(tt)
    print(t0)
    out.close()



# dataSet = landmarkDataset(txt_dir, img_dir)

from __future__ import print_function, division
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import urlparse
import urllib
import json
import math
import torch
from torch.autograd import Variable
import numpy as np
import argparse
import cv2

from PIL import Image, ImageDraw
import math
from skimage import io, transform, color
import pandas as pd
import os
import MyModel
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import MyUtils
#~ cv2.namedWindow('opencv')
predictionModel = ""
HOST_NAME = ''
PORT_NUMBER = 8010
imagePath = "uploadImages/"
use_gpu = 0
plt.ion()

def loadPytorchModel():
	corseNet = torch.load('output/' + "260baseLinecorseNet.pkl", map_location=lambda storage, loc:storage.cuda(use_gpu))
	fineNet = torch.load('output/' + "260baseLinefineNet.pkl", map_location=lambda storage, loc:storage.cuda(use_gpu))
	return corseNet, fineNet

def getCoordinate_new11(featureMaps, outputs2, lables, R1, R2, gpu, lastResult, coordinatesFine):


	#print("11", time.asctime())
	#~ landmarkNum = 33
	#~ imageNum = 1
	#~ featureNum = 33
	#~ featureMap = featureMaps[0]
	imageNum, featureNum, l, h, w = featureMaps[0].size()
	_, _, l_2, h_2, w_2 = outputs2.size()
	#~ print (imageNum, featureNum, l, h, w)
	landmarkNum = int(featureNum)


	#~ regionArea = R * R * math.PI
	
	#~ corse_landmark = getROIsFromHeatmap(featureMaps[0], R1, gpu).numpy()
	corse_landmark = lastResult.detach().cpu().numpy()
	#~ fine_landmark = getCoordinate_fine(outputs2, R2, gpu)
	
	#~ fine_landmark = getROIsFromHeatmap(outputs2, R2, gpu).numpy()
	fine_landmark = coordinatesFine.detach().cpu().numpy()


	X1, Y1, Z1 = np.round(corse_landmark[:, :, 0] * 767).astype('int'), np.round(corse_landmark[:, :, 1] * 767).astype('int'), np.round(corse_landmark[:, :, 2] * 575).astype('int')
	
	X2, Y2, Z2 = np.round(fine_landmark[:, :, 0] * h_2).astype('int'), np.round(fine_landmark[:, :, 1] * w_2).astype('int'), np.round(fine_landmark[:, :, 2] * l_2).astype('int')
	
	X_off, Y_off, Z_off = X2 - h_2//2, Y2 - w_2//2, Z2 - l_2//2
	
	GX, GY, GZ = np.round(lables[:, :, 0].numpy() * 767).astype('int'), np.round(lables[:, :, 1].numpy() * 767).astype('int'), np.round(lables[:, :, 2].numpy() * 575).astype('int')
	
	tot = torch.zeros((imageNum, landmarkNum, 3))
	for imageId in range(imageNum):
		for landmarkId in range(landmarkNum):
			
			x, y, z = X1[imageId][landmarkId], Y1[imageId][landmarkId], Z1[imageId][landmarkId]
			
			x_off, y_off, z_off = X_off[imageId][landmarkId], Y_off[imageId][landmarkId], Z_off[imageId][landmarkId]


			x_2 = x_off + x
			y_2 = y_off + y
			z_2 = z_off + z
			
			xx, yy, zz = GX[imageId][landmarkId], GY[imageId][landmarkId], GZ[imageId][landmarkId]
			
			
			tot[imageId][landmarkId] = torch.Tensor([x_2, y_2, z_2])

	return (tot)

def getScale(sourceImageH, sourceImageW):
	h = 769
	w = sourceImageW * h / sourceImageH
	return int(h), int(w)

def landmarkDetection(image_name):
	
	since = time.time()

	best_acc = [0, 0, 0, 0, 0, 0]
	
	test_avgOff = 0
	# Each epoch has a training and validation phase
	phase = 'val'
	#print ("1")
	corseNet.train(False)  # Set model to evaluate mode
	fineNet.train(False)
	#print ("2")

	# Iterate over data.

	img_name = "72_" + image_name + ".npy"
	img_name_origin = image_name + ".npy"
        
	image = np.load(img_name)
	image_origin = np.load(img_name_origin)
 	print (type(image))
	image = (image-np.mean(image))/ np.std(image)
	image_origin  = (image_origin-np.mean(image_origin))/ np.std(image_origin)
 	print (type(image))
	image  = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
	image_origin = torch.from_numpy(image_origin).float().unsqueeze(0).unsqueeze(0)
 	print (type(image))
	data = {'DICOM': image, 'DICOM_origin': image_origin}
	print (type(data['DICOM']))
 	print (type(image))
	inputs, inputs_origin = data['DICOM'].cuda(use_gpu), data['DICOM_origin']
 	print (inputs.size())
	heatMapsCorse, coordinatesCorse = corseNet(inputs)
	coordinatesCorse = coordinatesCorse.unsqueeze(0)

	ROIs = coordinatesCorse.cpu().detach().numpy()
   
	cropedtem = MyUtils.getcropedInputs(ROIs, inputs_origin, 64, -1)
	cropedInputs = [cropedInput.cuda(use_gpu) for cropedInput in cropedtem]
	data['cropedInputs'] = cropedInputs
				
	cropedInputs = data['cropedInputs']
	outputs2 = 0
	heatMapsFine, coordinatesFine = fineNet(ROIs, cropedInputs, outputs2)
	coordinatesFine = coordinatesFine.unsqueeze(0)
	print (coordinatesFine)
	coorall = coordinatesCorse + coordinatesFine
	#coorall = coordinatesCorse
	landmarks = MyUtils.getCoordinate_test3d(heatMapsCorse, heatMapsFine, use_gpu, coordinatesCorse, coordinatesFine) # getCoordinate_new1

	time_elapsed = time.time() - since
	print('test complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))

	return landmarks

class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
		try:
			self.send_response(200)
			self.send_header('Access-Control-Allow-Origin', '*')
			self.end_headers()
			response_string='Hello world'
			values=urlparse.parse_qs(urlparse.urlsplit(self.path).query)
			req=values['req'][0]
			data=json.loads(req)
			im_name=imagePath+data['imageUrl'].split('/')[-1].split('\\')[-1]
			urllib.urlretrieve(data['imageUrl'], im_name)
			
			land_marks = landmarkDetection(im_name)
			response_data={}
			imgid=1
			response_data['coordinates'] = []
			for landmark in land_marks:
				coordinate_data={}
				coordinate_data['landmarkId']='L'+str(imgid)
				coordinate_data['y'] = math.floor(landmark[1])
				coordinate_data['x']=math.floor(landmark[0])
				response_data['coordinates'].append(coordinate_data)
				imgid = imgid + 1

			response_data['apiKey'] = data['apiKey']
			response_data['imageId'] = data['imageId']

			response_string=json.dumps(response_data)
			
			self.wfile.write(bytes(response_string))
		except:
			self.wfile.write(bytes("server_error"))

if __name__ == '__main__':
    #~ server_class = HTTPServer
    #~ httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    corseNet, fineNet = loadPytorchModel()
    #TrainNet.test_model(corseNet, fineNet, dataloaders, criterion1, criterion2, optimizer_ft, epochs, use_gpu, R1, R2, saveName, landmarkNum, image_scale)
    
    landmarks = landmarkDetection("test")
    with open('landmarks.txt', 'w') as f:
		for landmark in landmarks:
			f.write(str(landmark[0]) + " " + str(landmark[1]) + " " + str(landmark[1]) + "\n" )
    '''
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
    '''


import os
import random
import shutil, os
import json
from PIL import Image
import PIL

def resizeImages():
	imagesFilePath = '/Users/keerthanakolisetty/Desktop/DrivingDataSubset/images/testing'
	resizedFilePath = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResized/images/testing/'

	for imageName in os.listdir(imagesFilePath):
			imagePath = os.path.join(imagesFilePath, imageName)
			originalImage = Image.open(imagePath)
			newImage = originalImage.resize((160, 90), PIL.Image.ANTIALIAS)
			newFileName = resizedFilePath + imageName
			newImage.save(newFileName)





def resizeLabels():
	labelsFilePath = '/Users/keerthanakolisetty/Desktop/DrivingDataSubset/labels/validation'
	resizedFilePath = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResized/labels/validation/'
	for labelName in os.listdir(labelsFilePath):
		labelPath = os.path.join(labelsFilePath, labelName)
		with open(labelPath) as f:
			originalData= json.load(f)
			resizedData = originalData
			for x in resizedData['labels']:
				x['box2d']['x1'] = x['box2d']['x1']/8
				x['box2d']['x2'] = x['box2d']['x2']/8
				x['box2d']['y1'] = x['box2d']['y1']/8
				x['box2d']['y2'] = x['box2d']['y2']/8

		resizedFilePath = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResized/labels/validation/' + labelName
		with open(resizedFilePath, 'w') as f:
		    json.dump(resizedData, f)
		    print('Written: ', labelName)

		
	
					



resizeImages()
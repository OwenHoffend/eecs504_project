import os 
import json
import shutil, os

def filterLabels():
	categories = ['car', 'truck', 'bus']
	labelsDirectory = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResized/labels/validation'

	labelsJsons = os.listdir(labelsDirectory)
	
	l = []
	for jsonLabel in labelsJsons:
		with open(os.path.join(labelsDirectory, jsonLabel)) as f:
			wholeLabel = json.load(f)
			res = list(filter(lambda i: i['category'] in categories, wholeLabel['labels']))
			wholeLabel['labels'] = res
			if(len(wholeLabel['labels'])!=0):
				l.append(wholeLabel)

	for x in l:
		name = x['name'][0:25]
		labelsSaveDirectory = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResizedFiltered/labels/validation/'+ name+".json"

		with open(labelsSaveDirectory, 'w') as f:
			json.dump(x, f)
			print('Written: ', name)


def filterImages(): 
	newImagesPath = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResizedFiltered/images/training/'
	labelsDirectory = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResizedFiltered/labels/training/'
	labelsFiles = os.listdir(labelsDirectory)
	for x in labelsFiles:
		name = x[0:25]
		imagePath = '/Users/keerthanakolisetty/Desktop/DrivingDataSubsetResized/images/training/' + name+'.jpg'
		shutil.copy(imagePath, newImagesPath)
		print('Copied: ', imagePath)


filterImages()
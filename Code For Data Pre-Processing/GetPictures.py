import os
import random
import shutil, os
import json



def getRandomImagesTraining():
    directory = '/Users/keerthanakolisetty/Desktop/DrivingDataFullset/images/train'
    trainingImages = []
    for filename in os.listdir(directory):
    	imageDir = os.path.join(directory, filename)
    	if(".DS_Store" not in imageDir):
    		imagesFiles = os.listdir(imageDir)
    		randomNumbers=[random.randint(1, len(imagesFiles)-1) for _ in range(6)]
    		for num in randomNumbers:
    			image = imagesFiles[num]
    			imagePath =  os.path.join(imageDir, image)
    			trainingImages.append(imagePath)

    for f in trainingImages:
        shutil.copy(f, '/Users/keerthanakolisetty/Desktop/DrivingDataSubset/images/training')
        print('Copied: ', f)



def getRandomImagesValidation():
    directory = '/Users/keerthanakolisetty/Desktop/DrivingDataFullset/images/val'
    validationImages = []
    for filename in os.listdir(directory):
        imageDir = os.path.join(directory, filename)
        if(".DS_Store" not in imageDir):
            imagesFiles = os.listdir(imageDir)
            randomNumbers=[random.randint(1, len(imagesFiles)-1) for _ in range(2)]
            for num in randomNumbers:
                image = imagesFiles[num]
                imagePath =  os.path.join(imageDir, image)
                validationImages.append(imagePath)

    for f in validationImages:
        shutil.copy(f, '/Users/keerthanakolisetty/Desktop/DrivingDataSubset/images/validation')
        print('Copied: ', f)


def getRandomImagesTesting():
    directory = '/Users/keerthanakolisetty/Desktop/DrivingDataFullset/images/test'
    testImages = []
    for filename in os.listdir(directory):
        imageDir = os.path.join(directory, filename)
        if(".DS_Store" not in imageDir):
            imagesFiles = os.listdir(imageDir)
            randomNumbers=[random.randint(1, len(imagesFiles)-1) for _ in range(2)]
            for num in randomNumbers:
                image = imagesFiles[num]
                imagePath =  os.path.join(imageDir, image)
                testImages.append(imagePath)

    for f in testImages:
        shutil.copy(f, '/Users/keerthanakolisetty/Desktop/DrivingDataSubset/images/testing')
        print('Copied: ', f)



def parseLabelsTraining():
    imagesDir = "/Users/keerthanakolisetty/Desktop/DrivingDataSubset/images/training"
    imagesFilesFullName = os.listdir(imagesDir)
    
    labelsDirectory = '/Users/keerthanakolisetty/Desktop/DrivingDataFullset/labels/train'
    labelsJsons = os.listdir(labelsDirectory)

    trainLabel = []
    for image in imagesFilesFullName:
        imageFileName = image[0:17]
        jsonFileName = imageFileName+".json"
        with open(os.path.join(labelsDirectory, jsonFileName)) as f:
            labels = json.load(f)
            for label in labels:
                if(label['name']==image):
                    trainLabel.append(label)


    for x in trainLabel:
        name = x['name'][0:25]
        fileLocation = '/Users/keerthanakolisetty/Desktop/DrivingDataSubset/labels/training/' + name+".json"

        with open(fileLocation, 'w') as f:
            json.dump(x, f)
            print('Written: ', name)


def parseLabelsValidation():
    imagesDir = "/Users/keerthanakolisetty/Desktop/DrivingDataSubset/images/validation"
    imagesFilesFullName = os.listdir(imagesDir)
    
    labelsDirectory = '/Users/keerthanakolisetty/Desktop/DrivingDataFullset/labels/val'
    labelsJsons = os.listdir(labelsDirectory)

    validationLabel = []
    for image in imagesFilesFullName:
        imageFileName = image[0:17]
        jsonFileName = imageFileName+".json"
        with open(os.path.join(labelsDirectory, jsonFileName)) as f:
            labels = json.load(f)
            for label in labels:
                if(label['name']==image):
                    validationLabel.append(label)


    for x in validationLabel:
        name = x['name'][0:25]
        fileLocation = '/Users/keerthanakolisetty/Desktop/DrivingDataSubset/labels/validation/' + name+".json"

        with open(fileLocation, 'w') as f:
            json.dump(x, f)
            print('Written: ', name)

getRandomImagesTesting()
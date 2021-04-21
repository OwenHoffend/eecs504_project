import cv2
import sys
import json
import matplotlib.pyplot as plt



def drawBoundingBoxes(imageData, labels):
    for label in labels:
        coordinate = label['box2d']
        left = int(coordinate['x1'])
        top = int(coordinate['y1'])
        right = int(coordinate['x2'])
        bottom = int(coordinate['y2'])
        imgHeight, imgWidth, _ = imageData.shape
        color = (0,255,0)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, 1)
        
        #if we want labels
        #label = label['category']
        #cv2.putText(imageData, label, (left, top - 12), 0, 1e-3 * imgHeight, color, 1)
        
    #plt.imshow(imageData)
    return imageData


def testBoundingBox():
    imagePath = "DrivingDataSubsetResized/images/training/03aa1144-5efc1938-0000047.jpg" 
    img = cv2.imread(imagePath)

    coordinates = []
    labelPath = "DrivingDataSubsetResized/labels/training/03aa1144-5efc1938-0000047.json"
    with open(labelPath) as f:
        labels= json.load(f)
        imageWithBox = drawBoundingBoxes(img, labels["labels"])
    
    cv2.imwrite("out.jpg", imageWithBox)

testBoundingBox()
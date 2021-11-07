import numpy as np
import cv2
import os as operatingSystem

rc = cv2.face.LBPHFaceRecognizer_create()
from PIL import Image

def getImagesAndLabels():
    allFaces=list()
    allIDs=list()
    for onePath in [operatingSystem.path.join('dataset',i) for i in operatingSystem.listdir('dataset')]:
        vectorizedImg=(np.array(Image.open(onePath).convert('L'),'uint8'))
        faces = cv2.CascadeClassifier("cascadeface.xml").detectMultiScale(vectorizedImg)
        for (left,right,up,down) in faces:
            allFaces.append(vectorizedImg[right:right+down,left:left+up])
            allIDs.append(int(operatingSystem.path.split(onePath)[-1].split(".")[1]))
    return allFaces,allIDs

print ("Training...\n")
allFaces,allIDs = getImagesAndLabels()
rc.train(allFaces, np.array(allIDs))
rc.write('trainer/model.yml')
print("Done\n")

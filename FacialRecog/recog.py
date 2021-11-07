import numpy as np
import cv2 as computerVision
import os as operatingSystem

model = computerVision.face.LBPHFaceRecognizer_create()
model.read('trainer/model.yml')

ID = 0
Names = list()
for i in ["None", "Rohan", "Temporary Individual", "ThisPersonDoesNotExist"]:
    Names.append(i)

webCam = computerVision.VideoCapture(0)
length = 640
height = 480
webCam.set(3,length)
webCam.set(4,height)

while True:
    ret, obtainedImage = webCam.read()
    for(left,right,up,down) in computerVision.CascadeClassifier("cascadeface.xml").detectMultiScale(computerVision.cvtColor(obtainedImage,computerVision.COLOR_BGR2GRAY), minNeighbors = 5, scaleFactor = 1.2, minSize = (int(0.1*webCam.get(3)), int(0.1*webCam.get(4)))):
        computerVision.rectangle(obtainedImage, (left,right), (left+up,right+down), (0,255,0), 2)
        ID, sure = model.predict(computerVision.cvtColor(obtainedImage,computerVision.COLOR_BGR2GRAY)[right:right+down,left:left+up])
        ID = Names[ID] if sure<100 else "unknown"
        computerVision.putText(obtainedImage, str(ID), (left+5,right-5), computerVision.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #computerVision.putText(obtainedImage,str(100-sure),(left+5,right+down-5),computerVision.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)   #Displaying confidence

    computerVision.imshow('webCam',obtainedImage)
    eK=computerVision.waitKey(10)
    if(eK==27):
        break
webCam.release()
computerVision.destroyAllWindows()

import os as operatingSystem
import cv2 as computerVision
import numpy
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import  load_model

model = load_model("model.h5")
webCam = computerVision.VideoCapture(0)

while True:
    ret, obtainedImage = webCam.read()
    if not ret:
        continue
    faces_detected = computerVision.CascadeClassifier("cascadeface.xml").detectMultiScale(computerVision.cvtColor(obtainedImage, computerVision.COLOR_BGR2RGB), 1.32, 5)

    for (left, right, up, down) in faces_detected:
        computerVision.rectangle(obtainedImage, (left, right), (left + up, right + down), (255, 0, 0), thickness=7)
        grayFace = computerVision.resize(computerVision.cvtColor(obtainedImage, computerVision.COLOR_BGR2RGB)[right:right + up, left:left + down], (224, 224))
        predictions = model.predict((numpy.expand_dims(img_to_array(grayFace), axis=0))/255)
        max_index = numpy.argmax(predictions[0])
        emotions = ('happy', 'neutral', 'sad','surprise')
        predicted_emotion = emotions[max_index]
        computerVision.putText(obtainedImage, predicted_emotion, (int(left), int(right)), computerVision.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    computerVision.imshow('Facial emotion analysis ', computerVision.resize(obtainedImage, (1000, 700)))
    if(computerVision.waitKey(10)==27):
        break
webCam.release()
computerVision.destroyAllWindows

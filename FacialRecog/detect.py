import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('cascadeface.xml')

noUsers = int(input("\nEnter the number of users you want to store images for "))

for i in range(noUsers):
    face_id = input('\nEnter user id end press <return> ==>  ')
    print("\nInitializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h,x:x+w])
            cv2.imshow('image', img)

        k = cv2.waitKey(10)
        if k == 27:
            break
        elif count >= 100: # Take 100 face sample and stop video
             break

    cam.release()
    cv2.destroyAllWindows()

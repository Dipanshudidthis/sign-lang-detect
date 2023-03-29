import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import gtts
import playsound
import os
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")  #it will compare the signs with the images in these parameters

offset = 20
imgSize = 300

folder = "Data/Yes"
counter = 0

labels = ["Hello","I Love You", "No", "Go", "Yes", "Like", "Promise", "All The Best", "Peace", "Wrong"]
ok=0
while True:
    ok+=1
    success, img = cap.read()
    imgOutput = img.copy()  #making copy of the original image
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)  #we are getting and index from the getprediction function
            #if index is 0 then the first element of labels array will be the sign
            #prediction is the confidence values that contains 3 elements in it

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False) #white image will not have any drawing

        print(ok)

        if ok%20==0:
            text = labels[index]
            sound = gtts.gTTS(text, lang="en")
            sound.save("sign1.mp3")
            playsound.playsound("sign1.mp3")
            os.remove("sign1.mp3")


        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED) #this rectangle will keep our text in a box
        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.QT_FONT_NORMAL, 1.7, (255, 255, 255), 2)  #we aee putting the labels of that specific index above our hand in the image and giving it color and font style
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        #rectangle will give our hand a box

        # cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

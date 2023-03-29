import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)  #adding our capture object, it initialises our webcam. '0' is the ID number of our webcam
detector = HandDetector(maxHands=1)  #detector will detect the hand

offset = 20
imgSize = 300

folder = "Data/Please"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  #hands are detected
    if hands:  #if hand is there
        hand = hands[0]  #because we only have one hand
        x,y,w,h = hand['bbox']  #we have x,y,width,height in boundingbox

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  #we created this image with white background to keep the dimensions consistent of portrait and landscape images
        #image is like a matrix so we can use numpy to create this image, imagesize=300x300 and '3' means it will be a colored image
        #matrix will have 0-255 (8 bit value), therefore np.uint8(unsigned integer 8 bit) and multiplying the values with 255 to get white background
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  #image is a matrix, so starting height is (y-offset) and ending height is y+h+offset, starting width is x-offset and ending width is x+w+offset. This will give us the exact bounding box required

        imgCropShape = imgCrop.shape  #this will overlay our image on white image, imgcrop is a matrix of 3 values, height,width,channel

        aspectRatio = h/w  #checking whether the height is bigger than the width

        #if height is bigger than width, we will stretch the height to 300, else we will stretch the width to 300
        #and then we will calculate the rest value and center the image to be in between the white image

        if aspectRatio>1:  #if height is greater than width
            k = imgSize/h
            wCal = math.ceil(k*w)   #calculating the width, and always round off to higher value
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)  #we are calculating the gap to fill to keep the image in between
            imgWhite[:, wGap:wCal + wGap] = imgResize #height is 300 since we did nothing, and starting width is the gap and ending width is width calculated + widthgap
            #now if we have a vertical image the height will always be 300 and width will be calculated

        else:   #if width is greater than height
            #logic is just the opposite of the above condition
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)  #showing our image
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  #waitkey will give one second delay
    if key == ord("s"):  #whenever we press s key, it will save the image
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite) #file name is this
        print(counter)

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "data/a"
counter = 0

while True:
    success, img = cap.read()
    hands,  img = detector.findHands(img) 
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize,3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        
        imgCropShape = imgCrop.shape


        aspectRatio = h/w
 
        if aspectRatio >1:
            c = imgSize/h
            cal = math.ceil(c*w)
            imgResize = cv2.resize(imgCrop,(cal, imgSize))
            imgResizeShape = imgResize.shape
            gap = math.ceil((imgSize-cal)/2)
            imgWhite[:, gap:cal+gap] = imgResize
        else:
            c = imgSize/w
            hcal = math.ceil(c*h)
            imgResize = cv2.resize(imgCrop,(imgSize, hcal))
            imgResizeShape = imgResize.shape
            hgap = math.ceil((imgSize-hcal)/2)
            imgWhite[hgap:hcal+hgap, :] = imgResize
            
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image",img )
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
    elif key == ord('q'):
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
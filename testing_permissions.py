import cv2
import numpy

"""
img = cv2.imread("Resources/pic.jpg") #read pic
cv2.imshow("Shubhan",img) #display pic
cv2.waitKey(0) #display for how long

"""
framewidth = 640
frameheight = 480
cap = cv2.VideoCapture(0)
cap.set(3, framewidth) #3 is the number set by opencv for width
cap.set(4, frameheight) #4 is the number set by opencv for width
while(True):
    success, img = cap.read()
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





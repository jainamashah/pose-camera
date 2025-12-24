from ultralytics import YOLO

import cv2 

import torch

import math

model = YOLO('yolo11m-pose.pt')

cap = cv2.VideoCapture(0)

print("Torch cuda is avail:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name())

def euclidean(x,y):
    return ((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5

def dot(ba,bc):
    return ba[0]*bc[0]+ba[1]*bc[1]

def mag(ba):
    return euclidean((0,0),ba)

def angle(a,b,c):
    ba = (b[0]-a[0],b[1]-a[1])
    bc = (b[0]-c[0],b[1]-c[1])
    ba_dot_bc = dot(ba,bc)
    mag_ba_bc = mag(ba)*mag(bc)
    return math.degrees(math.acos(ba_dot_bc/(mag_ba_bc+1e-6)))

while True:

    ret, img = cap.read()

    results = model(source=img,stream=False)

    for result in results:

        keypoints = result.keypoints.xy.cpu().numpy()[0]

        keypoints_normalized = result.keypoints.xyn.cpu().numpy()

        hands_up = str((angle(keypoints[8],keypoints[6],keypoints[12]) > 90) and (angle(keypoints[7],keypoints[5],keypoints[11]) > 90))

        img = result.plot()
        
        cv2.putText(img,("Chosen Mode: Hands Up click photo:"+ hands_up),(50,50)
            ,cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 0, 255), 3)
        
        cv2.putText(img,("Click photo:"+ hands_up),(50,100)
            ,cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 0, 255), 3)

        cv2.imshow("video",img)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



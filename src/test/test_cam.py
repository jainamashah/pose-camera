from ultralytics import YOLO

import cv2

model = YOLO('yolo11m-pose.pt')

cap = cv2.VideoCapture(0)

def classify_pose(keypoints):
    pass

while True:  

    results = model.predict(source=0,show = True, conf=0.3)

    for result in results: 
        keypoint = result.keypoints
        print(keypoint.xy.shape)

    # # Display the resulting image
    # cv2.imshow('Video', results)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    


import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture the first frame
ret, prev_frame = cap.read()

if not ret:
    print("Error: Couldn't read the first frame.")
    exit()

pre_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


# Mask to draw on
hsv_mask = np.zeros_like(prev_frame)
hsv_mask[:,:,1] = 255


while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(pre_frame, new_frame,None,0.5,3,15,3,5,1.2,0)
    mag,ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees = True)
    
    hsv_mask[:,:,0] = ang/2
    hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame',bgr)
    
    k = cv2.waitKey(10) & 0xFF
    
    if k == 27:
        break
    
    pre_frame = new_frame 
cap.release()
cv2.destroyAllWindows()
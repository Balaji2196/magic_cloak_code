import cv2
import time
import numpy as np
cap=cv2.VideoCapture(0)
time.sleep(3)
background=0

#capturing the background
for i in range(30):
    ret,background=cap.read()

background=np.flip(background,axis=1)

while(cap.isOpened()):
    ret,img=cap.read()

    if not ret:
        break
    img = np.flip(img, axis=1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #HSV values
    lower_red =np.array([0,120,70])
    upper_red =np.array([10,255,255])
    mask1=cv2.inRange(hsv,lower_red,upper_red) #seperating the cloak part

    lower_red =np.array([170,120,70])
    upper_red =np.array([180,255,255])
    mask2=cv2.inRange(hsv,lower_red,upper_red)

    mask1=mask1+mask2 #OR 1 0r x

    mask1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2) #noise removal,iteration=2 and it can be tuned for removing noise

    mask1=cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8),iterations=1) #dilate is used to smoothen the image

    mask2=cv2.bitwise_not(mask1) #except the cloak

    res1=cv2.bitwise_and(background,background,mask=mask1) #used for segmentation of the color
    res2=cv2.bitwise_and(img,img,mask=mask2) #used to substitute the cloak part
    final_output=cv2.addWeighted(res1,1,res2,1,0)
    cv2.imshow('final_output',final_output)
    k=cv2.waitKey(10)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
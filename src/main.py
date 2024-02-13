import cv2
import numpy as np

image = cv2.imread('img/test1.png')
image = cv2.resize(image,(1366,768))
img = image.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
factor = -50
toggle=0
while True:
    ret,thresh = cv2.threshold(gray,127+factor,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==5:
            cv2.drawContours(img,[cnt],0,(255,255,100),3)
        elif len(approx)==3:
            cv2.drawContours(img,[cnt],0,(0,255,0),3)
        elif len(approx)==4:
            cv2.drawContours(img,[cnt],0,(50,100,255),3)
    if toggle:
        cv2.imshow('img',img)
    else:
        cv2.imshow('img',image)

    key = cv2.waitKey(0)&0xff
    if key == ord('q') or key==0x1b:
        break
    if key == ord('w'):
        factor+=10
    if key == ord('s'):
        factor-=10
    if key == ord('a'):
        factor+=1
    if key == ord('a'):
        factor-=1
    if key==ord(' '):
        toggle^=1
    
    print("pressed",chr(key),"[factor]",factor)
cv2.destroyAllWindows()
import numpy as np
import cv2

def captureImage(resWidth = 1280, resHeight = 960, imageFileName = "capturedImage.jpg", doSave = True, show = True):

    cap = cv2.VideoCapture(1)
    cap.set(3, resWidth)
    cap.set(4, resHeight)

    # Capture frame-by-frame
    for i in range(10):
        ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # if show == True : 
    #     cv2.imshow("capturedImage", frame)
    #     # cv2.waitKey()

    if doSave == True :
        print(imageFileName)
        cv2.imwrite(imageFileName, frame)

    
    # When everything done, release the capture
    cap.release()
    return frame


img = captureImage()

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(9,9),8)
imgCanny = cv2.Canny(imgBlur,100,100)
cv2.imshow("Image Captured", img)
cv2.imshow("Image Captured Canny", imgCanny)
cv2.waitKey()
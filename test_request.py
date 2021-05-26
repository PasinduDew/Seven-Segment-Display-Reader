import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import requests

def console_log(message):
    print("" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " | " + message)



def preProcessImage(raw_image):

    imgGray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(9,9),8)

    # Cropping the Image to extract the Area Where the Digigit Appear
    # 1.2MP 1280x960
    croppedImage = imgBlur[325:648, 100:1198]
    # 0.9MP 1280x720
    # croppedImage = imgGray[205:525, 100:1198]
    # cv2.imshow("Cropped Image", croppedImage)

    # Thresholding the Image
    ret, threshImage = cv2.threshold(croppedImage, 100, 255, cv2.THRESH_BINARY)
    
    # Margings/ Bounderies of the Digits -> For Cropping
    marginList = [(0, 215), (225, 440), (450, 655), (680, 880), (890, threshImage.shape[1] )]

    rows = []

    # There are only 5 digits to extracted from the LCD Display
    for i in range(5):
        # print(marginList[i][0])
        digit = threshImage[:, marginList[i][0] : marginList[i][1]]

        # digitText = "Digit " + str(i)
        # cv2.imshow(digitText, digit)
        # cv2.imshow("Img" + str(i), digit)
        if isBlankDigit(digit) != True : 
            
            # print("Blank Digit Found")
            digit = cv2.resize(digit, (28, 28))
            # print(digit.shape)
            flatList = digit.flatten()
            newRow = []
            for val in flatList:
                newRow.append(val)
            rows.append(np.array(newRow, 'float').reshape(28, 28))
            
    return np.array(rows)

def captureImage(resWidth = 1280, resHeight = 960, imageFileName = "capturedImage.jpg", doSave = True, show = False):

    cap = cv2.VideoCapture(1)
    cap.set(3, resWidth)
    cap.set(4, resHeight)
    frame = None

    # Capture frame-by-frame
    for i in range(10):
        ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    if show == True : 
        cv2.imshow("capturedImage", frame)
        # cv2.waitKey()

    if doSave == True :
        print(imageFileName)
        cv2.imwrite(imageFileName, frame)

    
    # When everything done, release the capture
    cap.release()
    return frame


# To Check Whether the Area which is cropped is a blank area without a digit
def isBlankDigit(image):
    flatList = image.flatten()
    # print(flatList)
    noOfHighs = 0
    noOfLows = 0
    for val in flatList:
        if val > 250 :
            noOfHighs += 1
    # print(noOfHighs / flatList.size)
    if noOfHighs / flatList.size > 0.95 :
        return True
    else : 
        return False


def main():
    print("Capturing Image...")
    rawImage = captureImage()
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    _, img_encoded = cv2.imencode('.jpg', rawImage)
    r = requests.post('http://localhost:8000/api/v1/digit-prediction', headers=headers, data=img_encoded.tostring())
    print(r.json())

if __name__ == '__main__':
    main()

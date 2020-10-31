import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import cv2
import time

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

print(tf.__version__)

def read_csv_data(fileName):
    with open(fileName, 'r') as file:
        reader = csv.reader(file)
        tempArray = []
        for row in reader:
            tempArray.append(np.array(row, 'float').reshape(28, 28))
#         print
        return np.array(tempArray)

def read_csv_labels(fileName):
    with open(fileName, 'r') as file:
        reader = csv.reader(file)
        tempArray = []
        for row in reader:
#             print(row)
            tempArray.append(int(row[0]))
#         print
        return np.array(tempArray)


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

    cap = cv2.VideoCapture(0)
    cap.set(3, resWidth)
    cap.set(4, resHeight)

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


print("---------------------------------------------------------------------------------------------------------")
print("Loading Trained Model...")

loadBegin = time.time()

# Loading the Pre-Trained Model
model = tf.keras.models.load_model('ss_model')
print("Model Loaded Successfully --> Time Taken: %s s" %(time.time() - loadBegin))
# print("Execution Time: %s seconds" % (time.time() - startTime))
model.summary()
# x_test = read_csv_data("x_test_dataset.csv")
# print(x_test.shape)
# y_test = read_csv_labels("y_test_dataset.csv")

# x_test = x_test / float(x_test.max())
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Evaluate the model
# loss, acc = model.evaluate(x_test, y_test)
# print("Loaded model, accuracy: {:5.2f}%".format(100*acc))

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

print("Capturing Image...")

captureBegin = time.time()
rawImage = captureImage()

print("Image Captured Successfully  --> Time Taken: %s s" %(time.time() - captureBegin))
# cv2.imshow("Image", rawImage)

print("Pre Processing...")

numImageArray = preProcessImage(rawImage)
numImageArray = numImageArray / float(numImageArray.max())
numImageArray = numImageArray.reshape(numImageArray.shape[0], 28, 28, 1)

print("Digit Prediction Begins...")

predictBegin = time.time()
predictions = probability_model.predict(numImageArray)
numStr = ""
for prediction in predictions:
    numStr += str(np.argmax(prediction))

print("Digit Prediction Done --> Time Taken: %s s" %(time.time() - predictBegin))

print("-------------------------------------------------------------------------------------------------------------------")

numStr = str(numStr[:-2]) + "." + str(numStr[-2:])

print("Reading: " + numStr)
print("Total Time Taken: %s s" %(time.time() - loadBegin))

# cv2.waitKey(0)
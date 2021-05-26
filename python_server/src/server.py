from __future__ import print_function
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from fastapi import FastAPI, Request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import time
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from datetime import datetime




# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1xxhqVRrEB7QXOg0VENj5S_zZi7PDeC3GN79EemSzOwQ'
RANGE_NAME = 'A2'


def init_spreadsheet():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('sheets', 'v4', credentials=creds)

    return service
    
def read_csv_labels(fileName):
    with open(fileName, 'r') as file:
        reader = csv.reader(file)
        tempArray = []
        for row in reader:
#             print(row)
            tempArray.append(int(row[0]))
#         print
        return np.array(tempArray)

def console_log(message):
    print("" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " | " + message)


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


app = FastAPI()

# Loading the Pre-Trained Model
startTime = time.time()
console_log("Prediction Model Loading : ...")
model = tf.keras.models.load_model('../ss_model_v2')
model.summary()
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
console_log("Prediction Model Loading : Done")
console_log("Execution Time: %s seconds" % (time.time() - startTime))

service = init_spreadsheet()


@app.post("/api/v1/digit-prediction")
async def digit_prediction(request: Request):

    encoded_image = await request.body()

    # convert string of image data to uint8
    nparr = np.fromstring(encoded_image, np.uint8)
    # decode image
    rawImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    numImageArray = preProcessImage(rawImage)
    numImageArray = numImageArray / float(numImageArray.max())
    numImageArray = numImageArray.reshape(numImageArray.shape[0], 28, 28, 1)

    console_log("Digit Prediction Begins: ...")
    predictBegin = time.time()
    predictions = probability_model.predict(numImageArray)
    numStr = ""
    for prediction in predictions:
        numStr += str(np.argmax(prediction))

    console_log("Digit Prediction Begins: Done")
    console_log("Digit Prediction       : Time Taken: %s s" %(time.time() - predictBegin))

    numStr = str(numStr[:-2]) + "." + str(numStr[-2:])

    console_log("Reading: " + numStr)

    values = [
    [
        datetime.now().strftime("%d/%m/%Y %H:%M:%S"), numStr
    ],
    ]
    body = {
        'values': values
    }

    service.spreadsheets().values().append(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME, body=body, valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS").execute()
    
    return {"reading": numStr}
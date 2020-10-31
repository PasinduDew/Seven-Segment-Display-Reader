import cv2
import numpy
import os
import csv


path = 'E:\\Computer Vision\\Projects\\Seven-Segment-Recognition\\ImageSet\\RawTest_01'
dirTestImages = "./imageSet/testImages_05"


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




# -------------------------------------------------------------------------------------------------------------------------------------
#                                    Getting all the Original/Raw File Names in the Directory
# -------------------------------------------------------------------------------------------------------------------------------------
fileNames = []
# r=root, d=directories, f = files
for root, directories, files in os.walk(path):
    for file in files:
        if '.jpg' in file:
            # files.append(os.path.join(root, file))
            fileNames.append(os.path.join(root, file).split("\\")[-1])
            
            # print(os.path.join(root, file))


# -------------------------------------------------------------------------------------------------------------------------------------
#                                    Creating the Output Image Files and CSV - Dataset Creation
# -------------------------------------------------------------------------------------------------------------------------------------

# Output File Index -> Should Appemd At the End of the 'outputFileNamePrefix'
index = 0
# Prefix of the Output Image File
outputFileNamePrefix = "testImg_"
# Extension of the Output Image File
outputFileExtention = "jpg"

# CSV Rows
rows = []



for fileName in fileNames:
    
    rawImage = cv2.imread("./ImageSet/RawTest_01/" + fileName)
    # cv2.imshow(fileName, rawImage)

    greyImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grey Image", greyImage)

    # Cropping the Image to extract the Area Where the Digigit Appear
    # 1.2MP 1280x960
    croppedImage = greyImage[325:648, 100:1198]
    # 0.9MP 1280x720
    # croppedImage = greyImage[205:525, 100:1198]
    # cv2.imshow("Cropped Image", croppedImage)

    # Thresholding the Image
    ret,threshImage = cv2.threshold(croppedImage, 45, 255, cv2. THRESH_BINARY)
    # Margings/ Bounderies of the Digits -> For Cropping
    marginList = [(0, 215), (225, 440), (450, 655), (680, 880), (890, threshImage.shape[1] )]

# There are only 5 digits to extracted from the LCD Display
    for i in range(5):
        # print(marginList[i][0])
        digit = threshImage[:, marginList[i][0] : marginList[i][1]]

        # digitText = "Digit " + str(i)
        # cv2.imshow(digitText, digit)

        if isBlankDigit(digit) != True : 
            # print("Blank Digit Found")
            digit = cv2.resize(digit, (28, 28))
            # print(digit.shape)
            flatList = digit.flatten()
            newRow = [0]
            for val in flatList:
                newRow.append(val)
            rows.append(newRow)
            
            cv2.imwrite(dirTestImages +  "/" + outputFileNamePrefix + str(index) + "." + outputFileExtention, digit)
            index += 1


# -------------------------------------------------------------------------------------------------------------------------------------
#                                                       Writting to the CSV File
# -------------------------------------------------------------------------------------------------------------------------------------
with open('ss_dataset_train_02_Tresh45.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

cv2.waitKey()

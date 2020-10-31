import cv2
import numpy as np

rawImage = cv2.imread("test_01.jpg")
cv2.imshow("Image", rawImage)
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
    ret, threshImage = cv2.threshold(croppedImage, 70, 255, cv2.THRESH_BINARY)
    
    # Margings/ Bounderies of the Digits -> For Cropping
    marginList = [(0, 215), (225, 440), (450, 655), (680, 880), (890, threshImage.shape[1] )]

    rows = []

    # There are only 5 digits to extracted from the LCD Display
    for i in range(5):
        # print(marginList[i][0])
        digit = threshImage[:, marginList[i][0] : marginList[i][1]]

        # digitText = "Digit " + str(i)
        # cv2.imshow(digitText, digit)
        cv2.imshow("Img" + str(i), digit)
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

print(preProcessImage(rawImage).shape)

cv2.waitKey(0)
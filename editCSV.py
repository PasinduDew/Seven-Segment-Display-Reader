import cv2
import numpy
import os
import csv


def copyLabels(fromCSV, toCSV):
    fromLabelList = []
    toRows = []
    # Get the Lables from the CSV File
    with open(fromCSV, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            fromLabelList.append(row[0])
        # print(fromLabelList)

    with open(toCSV, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            toRows.append(row)


    for i in range(len(toRows)):
        temp = toRows[i][0]
        toRows[i][0] = fromLabelList[i]
        print(temp, " -> ", toRows[i][0])

    with open(toCSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(toRows)

    pass

def setLabels(inputCSVfileName= "",   labelList=[], deleteRowList=[], outputCSVfileName= ""):
    rows = []
    print("LabelList: ", len(labelList))
    print("DeleteList: ", len(deleteRowList))

    with open(inputCSVfileName, 'r') as file :
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)

    for i in range(len(rows)) :
        if i in deleteRowList :
            rows.pop(i)
            deleteRowList.remove(i)
            print("Before: ", deleteRowList)
            for j in range(len(deleteRowList)) :
                deleteRowList[j] = deleteRowList[j] - 1
            print("After: ", deleteRowList)
            
            pass

    for i in range(len(labelList)) :
        rows[i][0] = labelList[i]
            
        
    if outputCSVfileName == "" :
        outputCSVfileName = inputCSVfileName
        pass
    

    with open(outputCSVfileName, 'w', newline='') as file :
        writer = csv.writer(file)
        writer.writerows(rows)



# copyLabels("seven_segment_dataset_train_07.csv", "seven_segment_dataset_train_07_Thresh45.csv")

labels = [0, 0, 0, 1, 5, 5, 5,
          3, 4, 4, 4, 3, 4, 4,
          2, 3, 4, 4, 3, 1, 8,
          3, 0, 8, 1, 8, 5, 1,
          5, 1, 8, 5, 1, 1, 1,
          8, 3, 1, 0, 1, 8, 6,
          8, 9, 2, 7, 6, 5, 6,
          2, 9, 5, 4, 1, 2, 7,
          9, 9, 0, 2, 7, 9, 9,
          1, 2, 7, 9, 9, 3, 3,
          1, 1, 3, 6, 3, 2, 6,
          9, 1, 2, 1, 4, 5, 7,
          2, 1, 4, 5, 7, 1, 9,
          9, 0, 3, 2, 1, 4, 5,
          7, 6, 5, 9, 3, 6, 7,
          6, 0, 4, 8, 7, 2, 5,
          1, 0, 0, 6, 9, 8, 8,
          5, 4, 3, 4, 9, 5, 8,
          4, 1, 1, 1, 4, 9, 1,
          1, 1, 4, 8, 1, 0, 9,
          8, 1, 9, 0, 9, 1, 9,
          4, 2, 6, 7, 6, 0, 0,
          7, 7, 6, 7, 1, 0, 9,
          1, 1, 5, 7, 1, 1, 4,
          1, 5, 7, 6, 0, 4, 4,
          6, 4, 5, 8, 6, 5, 5,
          2, 6, 5, 5, 2, 6, 5,
          5, 1, 2, 5, 7, 5, 6,
          2, 5, 7, 0, 0, 2, 5,
          1, 6, 3, 2, 5, 9, 1,
          8, 2, 7, 8, 1, 1, 2,
          9, 3, 6, 5, 2, 9, 3,
          6, 3, 3, 2, 5, 0, 6,
          3, 2, 5, 0, 4, 2, 4,
          5, 5, 6, 2, 1, 6, 4,
          8, 2, 1, 4, 9, 3, 2,
          1, 6, 4, 9, 2, 4, 5,
          5, 9, 2, 4, 7, 7, 1,
          2, 1, 6, 7, 3, 2, 1,
          7, 6, 1, 2, 1, 5, 9,
          3, 2, 0, 6, 3, 2, 2,
          1, 6, 0, 0, 2, 1, 7,
          6, 6, 2, 2, 2, 7, 4,
          2, 2, 3, 2, 6, 2, 2,
          3, 4, 4, 2, 2, 9, 7,
          2, 2, 2, 9, 7, 9, 2,
          3, 0, 2, 8, 2, 3, 0,
          4, 3, 2, 3, 0, 3, 8,
          2, 3, 0, 4, 2, 2, 3,
          0, 4, 4, 2, 3, 0, 6,
          1, 2, 3, 0, 6, 0, 2,
          4, 9, 4, 3, 2, 4, 3,
          7, 5, 2, 3, 8, 1, 0,
          2, 4, 3, 5, 4]
deletes = []
setLabels(inputCSVfileName="ss_dataset_train_02_Tresh100.csv", labelList=labels, outputCSVfileName="ss_dataset_train_02_Tresh100_labeled.csv", deleteRowList=deletes)

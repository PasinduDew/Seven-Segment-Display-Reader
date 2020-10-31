
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
arrayVals = []
arrayImageData = []
with open('seven_segment_dataset_train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # print(row[0])
        arrayVals.append(row[0])
        arrayImageData.append(row[1 : ])

# print(arrayVals)
# print(arrayImageData)
# print(len(arrayImageData[0]))

with open('x_train_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(arrayImageData)

with open('y_train_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(arrayVals)

index = 1
img = np.array(arrayImageData[index], 'float32')
img = img.reshape(28, 28, 1)
print(img.shape)
cv2.imshow("Image: " + str(arrayVals[index]), img)
print(str(arrayVals[index]))
cv2.waitKey(0)
# plt.show()


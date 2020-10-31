import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

print(tf.__version__)

# Loading the Pre-Trained Model
model = tf.keras.models.load_model('ss_model')
model.summary()

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


x_test = read_csv_data("x_test_dataset.csv")
y_test = read_csv_labels("y_test_dataset.csv")

x_test = x_test / float(x_test.max())
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Evaluate the model
loss, acc = model.evaluate(x_test, y_test)
print("Loaded model, accuracy: {:5.2f}%".format(100*acc))

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)
print(predictions[6])
print(np.argmax(predictions[6]))
print(y_test[6])
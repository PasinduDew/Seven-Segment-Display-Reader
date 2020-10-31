import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

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


# from tensorflow.keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = read_csv_data("x_train_dataset.csv")
print(x_train.shape)
y_train = read_csv_labels("y_train_dataset.csv")
x_test = read_csv_data("x_test_dataset.csv")
y_test = read_csv_labels("y_test_dataset.csv")

x_train = x_train / float(x_train.max())
x_test = x_test / float(x_test.max())

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


inputShape = x_train[0].shape

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=64, kernel_size = (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0))

# Training
history = model.fit(x_train, y_train, batch_size=256, epochs=6, verbose=1, validation_data=(x_test, y_test))

print("################################ Saving... ###################################")
model.save('ss_model') 

# Evaluate the model
loss, acc = model.evaluate(x_test, y_test)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

def plot_learning_curve(history, epochs):
    # Plot Training and Accuracy Values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
    # Plot Training and Validation Loss Values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


plot_learning_curve(history, 6)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)
print(predictions[6])
print(np.argmax(predictions[6]))
print(y_test[6])


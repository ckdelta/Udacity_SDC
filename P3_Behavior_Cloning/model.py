from __future__ import print_function
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.regularizers import l2
import keras
import tensorflow as tf
from PIL import Image
from prepro import img_pre

#Parameters:
batch_size = 64
nb_epoch = 3

# input image dimensions
img_rows, img_cols = 80, 160
input_shape = (img_rows, img_cols, 3)

#Load data
x=list()
with open("driving_log0.csv") as f_handle:
#with open("data.csv") as f_handle:
    for line in f_handle.readlines():
        x.append(line.replace('\n', '').split(','))

#Load Image
d_size=len(x)
data=np.ndarray(shape=(d_size, 80, 160, 3), dtype=np.float32)
label=np.ndarray(shape=(d_size,1), dtype=np.float32)
c=0
for i in range(len(x)):
    if float(x[i][4])>0.7 and float(x[i][6])>15:
        image_t=x[i][0]
        #image_t='IMG/'+x[i][0][42:]
        load_img=Image.open(image_t)
        data[c,:,:,:]=img_pre(load_img)
        label[c,0]=x[i][3]
        c=c+1

real_size=c
dataset=np.ndarray(shape=(real_size, 80, 160, 3), dtype=np.float32)
labels=np.ndarray(shape=(real_size,1), dtype=np.float32)

for i in range(real_size):
    dataset[i,:,:,:]=data[i,:,:,:]
    labels[i,:]=label[i,:]

#Shuffle Dataset
permutation = np.random.permutation(real_size)
shuffled_dataset = dataset[permutation,:,:,:]
shuffled_labels = labels[permutation,:]
X_val=shuffled_dataset[0:1000,:,:,:]
Y_val=shuffled_labels[0:1000,:]
X_train=shuffled_dataset[1000:,:,:,:]
Y_train=shuffled_labels[1000:,:]
X_test=X_val
Y_test=Y_val

#repro Nvidia model
model = Sequential([
    Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", input_shape=input_shape),
    Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"),
    Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"),
    Convolution2D(64, 3, 3, subsample=(1, 1), activation="relu"),
    Convolution2D(64, 3, 3, subsample=(1, 1), activation="relu"),
    Flatten(),
    Dense(1164, activation='relu'),
#    Dropout(0.5),
    Dense(100, activation='relu'),
#    Dropout(0.5),
    Dense(50, activation='relu'),
#    Dropout(0.2),
    Dense(10, activation='relu'),
#    Dropout(0.2),
    Dense(1, activation='tanh'),
])

model.summary()

filepath='model.h5'
#model.load_weights(filepath)

#a=keras.optimizers.Adam(lr=0.0001)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])

print(X_train[0])
print(Y_train[0])
result = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, Y_val))

steering_angle = model.predict(X_test, batch_size=1000)
print(steering_angle[0:20])

#Save model
json_string = model.to_json()
with open ("model.json", "w") as json_file:
    json_file.write(json_string)
model.save_weights(filepath)

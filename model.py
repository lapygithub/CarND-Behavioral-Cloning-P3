#!/usr/bin/env python
#
# CarND - Behavioral Cloning P3
# Use the Nvidia CNN, use simulator training data to create trained model to drive the track
#
import os
import csv
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense
import time
from timeit import default_timer as timer

# Count the total seconds needed to train the model
start = timer()
print("Training Started: ",start)

# Read CSV containing image and steering angle data
samples = []
with open('./CarND-driving-data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# Split out train vs test 80/20
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

# Generator allows training data to be read in as needed for the batches
def generator(samples, batch_size=32):
	num_samples = len(samples)
	print ("Generator samples len=",num_samples)

	while 1: # Loop forever so the generator never terminates
			sklearn.utils.shuffle(samples)
			for offset in range(0, num_samples, batch_size):
				batch_samples = samples[offset:offset+batch_size]

				images = []
				angles = []
				for batch_sample in batch_samples:
					# read in images from center, left and right cameras
					path = "./CarND-driving-data/IMG/" # path to training IMG directory
					img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
					img_left = cv2.imread(path + batch_sample[1].split('/')[-1])
					img_right = cv2.imread(path + batch_sample[2].split('/')[-1])

					center_angle = float(batch_sample[3])

					# create adjusted steering measurements for the side camera images
					correction = 0.24 # this is a parameter to tune
					steering_left = center_angle + correction
					steering_right = center_angle - correction

					# add images and angles to data set
					images.append(img_center)
					images.append(img_left)
					images.append(img_right)
					angles.append(center_angle)
					angles.append(steering_left)
					angles.append(steering_right)

					# Flip
					images.append(np.fliplr(img_center)) # Center
					angles.append(-center_angle)
					images.append(np.fliplr(img_left)) # Left
					angles.append(-steering_left)
					images.append(np.fliplr(img_right)) # Right
					angles.append(-steering_right)

					# Shuffle the training data to randomize training every time
					X_train = np.array(images)
					y_train = np.array(angles)
					yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

row, col, ch = 160, 320, 3
input_shape = (row, col, ch)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(	lambda x: x/127.5 - 1.,
model.add(Lambda(	lambda x: (x / 255.0) - 0.5, input_shape=input_shape))

# set up cropping2D layer to remove horizon and car hood artifacts from input
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=input_shape)) # in-(160x320) -> out-(66x320)

# Implement the Nvidia CNN in Keras
model.add(Convolution2D(24,5,5,subsample=(2,2),border_mode="valid",activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),border_mode="valid",activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),border_mode="valid",activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(1,1),border_mode="valid",activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(1,1),border_mode="valid",activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Mean squared error loss and ADAM (A Method for Stochastic Optimization) optimizer
# ADAM Paper can be found here: http://arxiv.org/abs/1412.6980v8
model.compile(loss='mse', optimizer='adam')
# 4 epochs worked best, 5 overfit
model.fit_generator(train_generator,
										samples_per_epoch=len(train_samples), 
										validation_data=validation_generator,
										nb_val_samples=len(validation_samples), nb_epoch=4 )

model.save('model.h5')  # creates a HDF5 file 'model.h5'

# Number of seconds this run
end = timer()
print ("Model Saved. Elapsed time:",(end - start))


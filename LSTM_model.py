# IMPORT REQUIRED LIBRARIES

import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# from tensorflow import keras

# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import Nadam
# from tensorflow.keras.applications.vgg16 import VGG16

# from tensorflow.keras import models
# from tensorflow.keras.models import Sequential, Model 
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

# Set Numpy, Python and Tensorflow seeds to gconet consistent results.
seed_constant = 23
# np.random.seed(seed_constant)
# random.seed(seed_constant)
# tf.random.set_seed(seed_constant)


# Extract file
classes_list = ['PULL UPS', 'WEIGHTLIFTING', 'PUSH UPS', 'APPLYING MAKE UP', 'BOXING', 'LECTURING', 'PLAYING GOLF', 'MARCHING']
model_output_size = len(classes_list)

# Load model
model = tf.keras.models.load_model(r'LSTM_epochs_20_batch_64.h5')

seq_len = 70
img_height, img_width = 64, 64


# Predict Live model
def predict_on_live_video(video_file_path,seq_len, img_height, img_width):

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
    ####################################################################################
    
    dict_result = {}
    temp_list = []
    frames_list = []
    count = 0

    # Iterating through Video Frames
    while True:

        # Reading a frame from the video file 
        success, frame = video_reader.read() 
        count += 1

        # If Video frame was not successfully read then break the loop
        if not success or count > seq_len:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (img_height, img_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)

    while len(frames_list)<seq_len:
        frames_list.append(np.zeros((64,64,3),dtype = np.float32)) 

    ####################################################################################

    video_epx = np.expand_dims(frames_list, axis = 0)

    ####################################################################################

    # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
    predicted_labels_probabilities = model.predict(video_epx)[0]

    temp_list = predicted_labels_probabilities.copy().tolist()

    temp_list = sorted(temp_list,reverse=True)

    index0 = np.where(predicted_labels_probabilities==temp_list[0])[0][0]
    index1 = np.where(predicted_labels_probabilities==temp_list[1])[0][0]
    index2 = np.where(predicted_labels_probabilities==temp_list[2])[0][0]

    result0 = classes_list[index0]
    result1 = classes_list[index1]
    result2 = classes_list[index2]

    dict_result = {result0:predicted_labels_probabilities[index0],
                   result1:predicted_labels_probabilities[index1],
                   result2:predicted_labels_probabilities[index2]
                   }
                   
    for key, value in dict_result.items():
      dict_result[key] = float("{:.4f}".format(value))

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
    video_reader.release()

    return dict_result


# # PREDICTING

# video_title = '41 pull ups'

# # Getting the YouTube Video's path you just downloaded
# input_video_file_path = f'Youtube_Videos/{video_title}.mp4'

# # Calling the predict_on_live_video method to start the Prediction.
# predict_on_live_video(input_video_file_path, seq_len, img_height, img_width)

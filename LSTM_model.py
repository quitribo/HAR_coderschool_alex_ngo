################################################################################
#######################        IMPORT LIBRARY         ##########################
################################################################################

import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

################################################################################
#######################        GLOBAL SETTINGS        ##########################
################################################################################

# Define subcategories
classes_list = ['PULL UPS', 'WEIGHTLIFTING', 'PUSH UPS', 'APPLYING MAKE UP', 'BOXING', 'LECTURING', 'PLAYING GOLF', 'MARCHING']

# Define model output nodes
model_output_size = len(classes_list)

# Define input image height and width for mode
image_height, image_width = 64, 64
sequence_length = 70



################################################################################
#######################        LOAD MODEL             ##########################
################################################################################

# Load LSTM model
model = tf.keras.models.load_model(r'/models/LSTM_epochs_20_batch_64.h5')


################################################################################
#######################        MAKE PREDICTION         #########################
################################################################################


# Predict Live model
def predict_on_live_video(video_file_path,sequence_length, image_height, image_width):

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)
  
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
        if not success or count > sequence_length:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)

    while len(frames_list)<sequence_length:
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
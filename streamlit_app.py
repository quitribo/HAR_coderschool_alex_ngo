################################################################################
#######################        IMPORT LIBRARY         ##########################
################################################################################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import io

from LSTM_model import predict_on_live_video
from CNN_model import make_average_predictions

################################################################################
#######################        DESIGN INTERFACE       ##########################
################################################################################

st.set_page_config(
    page_title="PROJECT: HUMAN ACTIVITY RECOGNITION",
    layout='wide',
    initial_sidebar_state='auto',
)

choice = st.sidebar.radio('Select Models:',['CNN', 'VGG16 + LSTM'])


################################################################################
#######################        MAIN PROGRAM           ##########################
################################################################################

# DEFINE VGG16 + LSTM OPTION

if choice == 'VGG16 + LSTM':
    # Create box to input video
    st.title('Upload your videos (.mp4 / .mpeg)')
    uploaded_file = st.file_uploader(' ',['mp4','mpeg'])
    seq_len, img_height, img_width = 70,64,64

    # Adding columns to format video and prediction table
    col1,col2 = st.columns(2)

    if uploaded_file is not None:
        with col1:
            st.video(uploaded_file)

        with col2:            
                g = io.BytesIO(uploaded_file.read())            # BytesIO Object
                temporary_location = r"/test_videos/temp.mp4"     # save to temp file
                with open(temporary_location, 'wb') as out:     # Open temporary file as bytes
                    out.write(g.read())                         # Read bytes into file
                    out.close()                                 # close file
                input_video_file_path = temporary_location

                # Make prediction
                result = predict_on_live_video(input_video_file_path, seq_len, img_height, img_width)

                df = pd.DataFrame.from_dict(result, orient='index', columns=['Probability'])
                st.write(df)


# DEFINE CNN

elif choice == 'CNN':
    st.write('')
    # Create slider bar
    frame_rate = st.sidebar.slider(
        "Number of Frames",
        min_value=1,
        max_value=100,
        value=100,
        step=1,
        help="Number of frames that CNN uses to average",
    )

    # Create box to input video
    st.title('Upload your videos (.mp4 / .mpeg)')
    uploaded_file = st.file_uploader(' ',['mp4','mpeg'])
    image_height, image_width = 64,64
    predictions_frames_count = frame_rate

    # Adding columns to format video and prediction table
    col1,col2 = st.columns(2)

    if uploaded_file is not None:
        with col1:
            st.video(uploaded_file)

        with col2:
                g = io.BytesIO(uploaded_file.read())            # BytesIO Object
                temporary_location = r"/test_videos/temp.mp4"     # save to temp file
                with open(temporary_location, 'wb') as out:     # Open temporary file as bytes
                    out.write(g.read())                         # Read bytes into file
                    out.close()                                 # close file
                input_video_file_path = temporary_location

                # Make prediction
                result = make_average_predictions(input_video_file_path, predictions_frames_count)

                df = pd.DataFrame.from_dict(result, orient='index', columns=['Probability'])
                st.write(df)

    

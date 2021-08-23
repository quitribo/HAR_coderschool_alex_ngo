
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from LSTM.py import *


st.set_page_config(
    page_title="PROJECT: HUMAN ACTIVITY RECOGNITION",
    layout='wide',
    initial_sidebar_state='auto',
)

choice = st.sidebar.radio('Select Models:',['CNN', 'VGG16 + LSTM'])

if choice == 'CNN':
    st.title('Puppy can display images!')
    photo_uploaded = st.file_uploader('Choose your best puppy photo',['png','jpg','jpeg'])
    

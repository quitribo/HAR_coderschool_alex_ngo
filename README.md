# PROJECT : HUMAN ACTIVITY RECOGNITION    
_Trung Ngo_

_Coderschool MLE part-time 08/2021_


## Summary

This project is about Human Activity Recognition. Project is completed in 4 weeks under the supervision of Coderschool, VN.

2 methods were implemented:
* __CNN__
    * Rolling average from 1 to 100 frames per video prediction. 
    * Trained with 15 epochs & batch of 16
    
* __VGG16 + LSTM__
	* VGG16 is used as transfer learning to extract features
	* LSTM (timestep = 256)
	* Trained with 20 epochs & batch of 64
	
    

|  | Description |
| --- | ----------- |
| **Problem type** | Video Classification |
| **Approaches** | CNN & Transfer Learning with LSTM |
| **Dataset** | **[Modified UFC101](https://www.crcv.ucf.edu/data/UCF101.php)** |
| **Theory** | Basic Convolutional Neural Network |
|  | Recurrent Neural Network |
|  | Transfer Learning |
| **Duration** | 4 weeks |
| **Platform** | Streamlit app |
| **Final Product** | **[github](https://github.com/quitribo/coderschool_final_project)** |


## Github repo contents
* **code**  
    - CNN_model.py 
    - LSTM_model.py 
    - app.py 
* **models**
    - CNN_epochs_15_batch_16.h5
    - LSTM_epochs_20_batch_64.h5
* **presentation**
    - HAR_Coderschool_08_2021_Trung_Ngo
* **test_videos**
    - videos x 7



## General info



## Prepare Dataset:

__1. Classes_list__
> __Note:__
> Original UCF101 dataset is a collection of realistic action videos from YouTube, with 13320 videos from 101 action categories in total.
> 
> Due to its large size (~7gb), only 8 categories were selected for training. Videos of similar actions are grouped together to make a bigger dataset for every category.  Roughly 200-250 videos (30fps) per each subgroup, streamlined to keep videos that are more than 3s



![classes_list](https://i.imgur.com/Zv5t8kL.png)


__2. Extract, Resize and Normalize Frames__

I created a function called __frames_extraction(video_path)__ that extracts frames from each video. 
#### Here’s how this function works:

- It takes a video file path as input.
- It then reads the video file frame by frame.
- Resizes each frame
- Normalizes the resized frame
- Appends the normalized frame into a list
- Finally returns that list.


:bulb: **Note:** function written for CNN with modification for LSTM

```python
def frames_extraction(video_path):
    # Empty List declared to store video frames
    frames_list = []
    
    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)

    # Iterating through Video Frames
    while len(frames_list) < max_images_per_class:

        # Reading a frame from the video file 
        success, frame = video_reader.read() 

        # If Video frame was not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)

    # Closing the VideoCapture object and releasing all resources. 
    video_reader.release()

    # returning the frames list 
    return frames_list
```


__3. Dataset Creation__

I created a function called __create_dataset()__, this function uses the frame_extraction() function above and creates the final preprocessed dataset.

#### Here’s how this function works:
- Iterate through all the classes in classes_list
- Iterate through all the video files present in each class.
- Call the frame_extraction method on each video file.
- Add the returned frames to a list called temp_features
- After all videos of a class are processed, randomly select video frames `(equal to max_images_per_class)` and add them to the list called features.
- Add labels of the selected videos to the `labels` list.
- After all videos of all classes are processed then return the features and labels as NumPy arrays.


:bulb: **Note:** function written for CNN with modification for LSTM
``` python
def create_dataset():

    # Declaring Empty Lists to store the features and labels values.
    temp_features = [] 
    features = []
    labels = []
    
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        if class_name != '.ipynb_checkpoints':
            print(f'Extracting Data of Class: {class_name}')
            
            # Getting the list of video files present in the specific class name directory
            files_list = os.listdir(os.path.join(dataset_directory, class_name))

            # Iterating through all the files present in the files list
            for file_name in files_list:

                # Construct the complete video path
                video_file_path = os.path.join(dataset_directory, class_name, file_name)

                # Calling the frame_extraction method for every video file path
                frames = frames_extraction(video_file_path)

                # Appending the frames to a temporary list.
                temp_features.extend(frames)
            
            # Adding randomly selected frames to the features list
            features.extend(random.sample(temp_features, max_images_per_class))

            # Adding Fixed number of labels to the labels list
            labels.extend([class_index] * max_images_per_class)
            
            # Emptying the temp_features list so it can be reused to store all frames of the next class.
            temp_features.clear()
        else:
            break

    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.asarray(labels)  

    return features, labels
```


Calling the create_dataset method which returns features and labels.

- For CNN : 4D tensor

![](https://i.imgur.com/JtCUMMj.png)

- For LSTM : 5D tensor

![](https://i.imgur.com/VfcIggc.png)

__4. Create tensor slice to split train, test set__

This prevents model from crashing by limiting how much to feed in model during training.

![](https://i.imgur.com/0u3hs3w.png)




## Train with CNN model

:rocket: __How CNN model works__

CNN model does not look at the entire video sequence but just classifying each frame independently. A solution to this problem is to average the prediction result over 5,10, or n extracted frames from the video.

![CNN](https://learnopencv.com/wp-content/uploads/2021/01/Predictions-on-the-sequence-of-frames-of-a-video-of-a-person-running-1024x734.jpg)

:rocket: __Model architecture__

![CNN](https://i.imgur.com/cgRF6P2.png)


:rocket: __Training with 15 epochs and batch_size = 16__

![](https://i.imgur.com/ecqm1t2.png)

:rocket: __Plot Model’s Loss and Accuracy Curves__

![](https://i.imgur.com/7dmCCgb.png)


:rocket: __Using Single-Frame CNN Method__

Created a function called `make_average_predictions(video_file_path, predictions_frames_count)` that takes `n` frames from the entire video and make predictions. In the end, it will average the predictions of those n frames to give us the final activity class for that video. `n` is set as variable in streamlit.

Expected Result

![](https://i.imgur.com/vx3g2Vh.png)




## Train with VGG16 + LSTM model

:rocket: __How VGG16 + LSTM model works__

The idea in this approach is to use pre-trained VGG16 to extract local features of each frame on the input data, transforming it from pixels input into an internal matrix

The outputs of this VGG16 are then fed to a multilayer LSTM network. In my model, the CNN is not trained. It acts more like an image interpretation for my LSTM model.


![](https://i.imgur.com/ICShQOR.png)

[Source](https://www.semanticscholar.org/paper/CNN-LSTM-Architecture-for-Action-Recognition-in-Orozco-Buemi/df8beecc6c0d16e9b75675c46b99aee80aaa83d5)

:rocket: __Model architecture__

![](https://i.imgur.com/V3MZTK1.png)



:rocket: __Training with 20 epochs and batch_size = 64__

I achieved a very high accuracy on both training and validation set (~92% on validation and ~97% on training set).



:rocket: __Plot Model’s Loss and Accuracy Curves__

![](https://i.imgur.com/IHSaddq.png)

:rocket: __Prediction__

Created a function called `predict_on_live_video(video_file_path, seq_len, img_height, img_width)` could make prediction on entire video. 

`seq_len, img_height, img_width` are 70 frames per video, input image size of 64x64

Expected result

![](https://i.imgur.com/vBF0cnN.png)


## Streamlit app


![](https://i.imgur.com/c4atNyp.png)




## Running app
Download this repository and test with videos in "test_videos" folder:

```
$ cd ../code
$ streamlit run app.py
```


## References

- Reference code

https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/

- Tutorial - Building video classification model

https://www.analyticsvidhya.com/blog/2019/09/step-by-step-deep-learning-tutorial-video-classification-python/

- Recurrent Neural Networks

https://www.youtube.com/watch?v=WCUNPb-5EYI&ab_channel=BrandonRohrer

- Papers for long-short term memory

https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
https://thebinarynotes.com/video-classification-keras-convlstm/

- Tutorial - using Keras to build LSTM

https://riptutorial.com/keras/topic/9656/dealing-with-large-training-datasets-using-keras-fit-generator--python-generators--and-hdf5-file-format

- Tutorial - building app with Streamlit

https://towardsdatascience.com/human-pose-estimation-for-baseball-swing-using-opencv-and-openpose-74d3a109c454

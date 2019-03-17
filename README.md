# Behavioral-Cloning
# End to End Learning for Self-Driving Cars

## Overview

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Dependencies

This lab requires:
* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here]
(https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Project structure

* drive.py - Implements driving simulator callbacks, essentially communicates with the driving simulator app providing model predictions 
             based on real-time data simulator app is sending.
* model.py - Containing the script to create and train the model
* video.py - Using to create the video recording when in autonomous mode.
* model_my.h5 - Model weights.
* run1.mp4 - The video recording when in autonomous mode.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## Implementation
### Model Architecture and Training Strategy

#### An appropriate model architecture has been employed

The Model Architecture was published by the autonomous vehicle team at NVIDIA. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
Firstly, the number of epochs is 30.I found that the loss of training set and validation set has been going down. Then I set the number of epochs is 50.The model consists of three convolution neural network with 5x5 filter sizes and depths are 12,24,36. There are also two convolution neural network with 3x3 filter sizes and depths are both 64.Then the model has a flatten layer and four fully connected layer with the depths are 100,50,10,1.The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer .What’s more, the images resize to 66*200 and make cropping using a Keras lambda layer.
![](https://github.com/Luzhongyue/Behavioral-Cloning/blob/master/Images/NVDIA-model.png)

#### Appropriate training data

Training data comes from https://github.com/shawshany/behavioral-clone/tree/master/data. The data has a combination of center lane driving, recovering from the left and right sides of the road.(My computer’s GPU is so bad to support I use the simulator to collect data, the choppy animation make a big influence for collecting data.)

#### Creation of the Training Set & Training Process

I randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 . I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Data Augmentation

For this project, recording recoveries from the sides of the road back to center is effective. But it is also possible to use all three camera images to train the model. So I feed the left and right camera images to your model as if they were coming from the center camera. This way, I can teach the model how to steer if the car drifts off to the left or the right. After the collection process, I had about 19254 number of data points.

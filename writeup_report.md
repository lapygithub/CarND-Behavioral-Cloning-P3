# **Behavioral Cloning Project** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NVIDIA_End_to_End_CNN.png "Model Visualization"
[image2]: ./examples/center_2017_09_16_14_46_54_103_centered.jpg "Center Driving"
[image3]: ./examples/center_2017_09_16_14_50_23_285_reverse.jpg "Reverse Driving"
[image4]: ./examples/center_2017_09_17_09_28_49_193_recov1.jpg "Recovery Image"
[image5]: ./examples/center_2017_09_17_09_28_50_657_recov2.jpg "Recovery Image"
[image6]: ./examples/center_2017_09_17_09_28_51_238_recov3.jpg "Recovery Image"
[image7]: ./examples/placeholder_small.png "Normal Image"
[image9]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model used the NVIDIA architecture as first described [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  It consists of 3 convolution neural network (CNN) with 5x5 filter sizes and depths of 24, 36 and 48, followed by 2 CNN with 3x3 filters and depths of 64. (model.py lines 95-99) The model then flattens and implements 4 fully connected layers of depth 1164, 100, 50, 10 and finally outputs at 1. (model.py lines 100-105)

The model includes RELU layers to introduce nonlinearity in all convolutions, and the data is normalized in the model using a Keras lambda layer (code line 89). 

#### 2. Attempts to reduce overfitting in the model

The NVIDIA model did not describe using dropout layers in order to reduce overfitting and I have confirmed it successfully without it.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80-81 and 111). I found that running too many epochs was overfitting after 4 runs.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).

#### 4. Appropriate training data

I used many different approaches to secure enough training data.
See the 1. Solution design approach for data details and driving attempts.  

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to go straight to the NVIDIA model, implement training data and then provide more and more training data.

My first step was to use the NVIDIA convolution neural network model based on their success instead of going throught the progression of models in the lesson.  Data augmentation seemed more important than experimenting with substandard models.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20).

I did not add any dropout or pooling layers for overfitting.

I went through multiple attempts to create the proper training data to keep the vehicle driving on the road. First step was to try center lane driving which was difficult due to the simulator lagging response time.  Turning the simulator to the lowest resolution and fastest speed allowed for better manual driving (I ended up using the mouse to steer and W-key for acceleration. First attempt with this data simply drove off the road.
Next, I pulled in the Udacity training set with no better results.
At this point I added left and right camera views, flip horizontally and to add more training data, drove the track backward a and forward a few more times.  This allowed the autonomous mode to drive the curves, but the sharp curves were still an issue.
Finally, I added lots of recovering from the left and right sides of the road and drove the sharp curves multiple times, again backwards and forwards.

Initially, I found that my model was trending towards more error when going to 5 epochs so I settled on 4 epochs.

4 epoch run of the training and validation:
    [mikel@MikeL-Mac CarND-Behavioral-Cloning-P3]$ python model.py
    Using TensorFlow backend.
    Training Started:  809768.27625151
    Generator samples len= 16064
    Generator samples len= 4017
    Epoch 1/4  16110/16064 - 399s - loss: 0.0290 - val_loss: 0.0416
    Epoch 2/4  16116/16064 - 386s - loss: 0.0297 - val_loss: 0.0358
    Epoch 3/4  16110/16064 - 370s - loss: 0.0300 - val_loss: 0.0396
    Epoch 4/4  16164/16064 - 365s - loss: 0.0359 - val_loss: 0.0454
    Model Saved. Elapsed time: 1530.5871465320233

At the end of the this process, the vehicle is able to drive autonomously around the track without leaving the road.

But in writing up the report I found an error in my CNN, where mistakenly, the fully connected layer that was supposed to be 50 was 500!  I also was debating the need for a fully connected 1164 layer to mimic NVIDIA's diagram after the flatten.  I'm still not understanding how the flatten of a 64@1x18 input is equal to 1164 in their diagram.  My calculator has 64x1x18=1152 not 1164.  For the final run, I opted to leave out the 1164 full connected layer and just run the output of my flatten (22x64=2112) into the final fully connected layers (100,50,10).

#### 2. Final Model Architecture

The final model architecture (model.py lines 86-105) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 | Description                                   | 
|:----------------------|:----------------------------------------------| 
| Input                 | 160x320x3 Simulator image                     | 
| Lambda Normalization  | output 160x320x3                              | 
| Cropping2D            | Crop (70,24), output 66x320x3                 | 
| L1:Convolution 24,5x5 | 2x2 stride, 'valid' padding, output 31x158x24 | 
| L1:RELU               | Activation                                    |
(W−F+2P)/S+1
x:(66-5)/2 + 1 = 61/2 + 1 = 31, y:(320-5)/2 + 1 = 315/2 + 1 = 158
| L2:Convolution 36,5x5 | 2x2 stride, 'valid' padding, output 14x77x36  | 
| L2:RELU               | Activation                                    | 
x:(31-5)/2 + 1 = 26/2 + 1 = 14, y:(158-5)/2 + 1 = 153/2 + 1 = 77 
| L3:Convolution 48,5x5 | 2x2 stride, 'valid' padding, output 5x37x48   | 
| L3:RELU               | Activation                                    | 
x:(14-5)/2 + 1 = 9/2 + 1 = 5, y:(77-5)/2 + 1 = 72/2 + 1 = 37 
| L4:Convolution 64,3x3 | 1x1 stride, 'valid' padding, output 3x35x64   | 
| L4:RELU               | Activation                                    | 
x:(5-3)/2 + 1 = 2/1 + 1 = 3, y:(37-3)/2 + 1 = 34/1 + 1 = 35  
| L5:Convolution 64,3x3 | 1x1 stride, 'valid' padding, output 1x33x64   | 
| L5:RELU               | Activation                                    | 
x:(3-3)/2 + 1 = 0/1 + 1 = 1, y:(35-3)/2 + 1 = 32/1 + 1 = 33  
| L6:Flatten            | output 2112                                   | 
| L7:Fully connected    | output 100                                    | 
| L8:Fully connected    | output 50                                     | 
| L9:Fully connected    | output 10                                     | 

Here is a visualization of the architecture directly from the [NVIDIA paper] (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first figured out that I couldn't drive at high quality and high resolution simulator mode and reset the image to low resolution and fastest rendering quality.  I then recorded at least two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle driving the track in reverse.  Here is an example of reverse driving:

![alt text][image3]


I did some work on recovering from the left side and right sides of the road back to center so that the vehicle would learn to coarse correct. These images show what a recovery looks like starting from the left side of the road :

![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data sat, I flipped images and angles thinking that this would double the training data and train for left and right curves simultaneously. Flipping was done using numpy flip.  (See model.py lines 66-72)

After the collection process, I had 20081x2(flip)x3(left/right)=120486 number of data points. While the images could have been preprocessed outside of the model, in this case I used inline normalization (lambda) and horizontal cropping.  Cropping removed extraneous horizon artifacts at the top and the hood of the car at the bottom.

The data set was randomly shuffled and 20% of the data was used in the validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by 5 epochs or greater overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

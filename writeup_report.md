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

My model used the NVIDIA architecture as first described [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  It consists of 3 convolution neural network (CNN) with 5x5 filter sizes and depths of 24, 36 and 48, followed by 2 CNN with 3x3 filters and depths of 64. (model.py lines 100-111) The model then flattens and implements 4 fully connected layers of depth 1164, 100, 50, 10 and finally outputs at 1. (model.py lines 112-117)

The model includes RELU layers to introduce nonlinearity in all convolutions, and the data is normalized in the model using a Keras lambda layer (code line 95). 

#### 2. Attempts to reduce overfitting in the model

The NVIDIA model did not describe using dropout layers in order to reduce overfitting and I have confirmed it converges successfully without it.  I had more trouble running a higher number of epochs and settled on 4.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 85-87 and 117-120).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

I used many different approaches to secure enough training data.
First round of data collection:
    * I went through multiple attempts to create the proper training data to keep the vehicle driving on the road. First step was to try center lane driving which was difficult due to the simulator lagging response time.  Turning the simulator to the lowest resolution and fastest speed allowed for better manual driving (I ended up using the mouse to steer and W-key for acceleration. First attempt with this data simply drove off the road.
    * Next, I pulled in the Udacity training set with no better results.
    * At this point I added left and right camera views, flip horizontally and to add more training data, drove the track backward a and forward a few more times.  This allowed the autonomous mode to drive the curves, but the sharp curves were still an issue.
    * Finally, I added lots of recovering from the left and right sides of the road and drove the sharp curves multiple times, again backwards and forwards.
    * Ultimately, I scrapped this data set and started over as explained in 1. Solution Design Approach
Second round of data collection:
    * Centerline drive the track smooth as possible for two laps.
    * Copy in the Udacity driving data.
    * Centerline drive the two sharp turns (left and right) twice as smooth as possible.  I actually drove these really slow and smooth.
    * Make 10+ copies of the sharp turn entries in the CSV (This helps to normalize the number of examples of sharp turn driving versus large radius turn driving).
    * Add a lap of edges to center corrections.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to go straight to the NVIDIA model, implement training data and then provide more and more training data.

My first step was to use the NVIDIA convolution neural network model based on their success instead of going throught the progression of models in the lesson.  Data augmentation seemed more important than experimenting with substandard models.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20).

I did not add any dropout or pooling layers for overfitting.

Initially, I found that my model performed worse when configuring 5+ epochs so I settled on 4 epochs.

See 4. Appropriate Training Data for details on driving attempts.

At the end of the this process, the vehicle was able to drive autonomously around the track without leaving the road. See video_run1.mp4.

But in writing up the report, I found an error in my CNN, where mistakenly, the fully connected layer that was supposed to be 50 was 500!  I also was debating the need for a fully connected 1164 layer to mimic NVIDIA's diagram after the flatten.  I'm still not understanding how the flatten of a 64@1x18 input is equal to 1164 in their diagram.  My calculator has 64x1x18=1152 not 1164.

After setting the NVIDIA fully connected layer back to 50, driving autonomously failed.  Setting it back to 500 failed too. After multiple attempts at retraining, I decided to start over with better training data as my first success must have been due to randomly picking a good portion of the data to train on.  I also knew that there were plenty of driving mistakes in the first set of training data.

At this point of correctly configuring the NVIDIA fully connected layers, I decided to also try to match the NVIDIA image size of 66 pixels by cropping off the bottom 24 pixels.  This model would not drive the car on the track!
I decided to leave the image size at 70 pixels.

Again, at the end of the this process, the vehicle is able to drive autonomously around the track without leaving the road.   See video_run2.mp4.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-117) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 | Description                                   | 
|:----------------------|:----------------------------------------------| 
| Input                 | 160x320x3 Simulator image                     | 
| Lambda Normalization  | output 160x320x3                              | 
| Cropping2D            | Crop (70,20), output 70x320x3                 | 
| L1:Convolution 24,5x5 | 2x2 stride, 'valid' padding, output 33x158x24 | 
| L1:RELU               | Activation                                    |
| L2:Convolution 36,5x5 | 2x2 stride, 'valid' padding, output 15x77x36  | 
| L2:RELU               | Activation                                    | 
| L3:Convolution 48,5x5 | 2x2 stride, 'valid' padding, output 6x37x48   | 
| L3:RELU               | Activation                                    | 
| L4:Convolution 64,3x3 | 1x1 stride, 'valid' padding, output 4x35x64   | 
| L4:RELU               | Activation                                    | 
| L5:Convolution 64,3x3 | 1x1 stride, 'valid' padding, output 2x33x64   | 
| L5:RELU               | Activation                                    | 
| L6:Flatten            | output 4224                                   | 
| L7:Fully connected    | output 1164                                   | 
| L8:Fully connected    | output 100                                    | 
| L9:Fully connected    | output 50                                     | 
| L10:Fully connected   | output 10                                     | 
| L11:Fully connected   | output 1                                      | 

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

To augment the data sat, I flipped images and angles thinking that this would double the training data and train for left and right curves simultaneously. Flipping was done using numpy flip.  (See model.py lines 72-78)

For the first collection process, I had 20081x2(flip)x3(left/right)=120486 number of data points. The second set had (19377+4845=24222)x2(flip)x3(left/right)=145332 data points.

While the images could have been preprocessed outside of the model, in this case I used inline normalization (lambda) and horizontal cropping.  Cropping removed extraneous horizon artifacts at the top and the hood of the car at the bottom.  If I rework this model in the future, I'd like to try resizing to 66x200 to exactly match the NVIDIA input image and then remove the 1164 fully connected layer.

The data set was randomly shuffled and 20% of the data was used in the validation set. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 3. Conclusion
This project taught me the importance of collecting only very good training data.  Having tools to visualize and analyze the driving runs would have been very useful to find a remove bad driving sections.  For example, a jerk of the hand on the mouse in a turn which sends the car opposite the curve.
Having a joystick would have helped driving too.

I still find it amazing that CNN pick out features from an image and end to end, predict steering angle correctly.


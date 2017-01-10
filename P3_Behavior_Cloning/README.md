# Project3: Behavior Cloning

In this project, the ultimate purpose is to train a DNN to automatically drive a car within road region on the simulator provided. To acheive the goal, three sections will be illustrated below: 1) Recording data 2) Model architecture 3) Training process.

## Table of Contents

1. [Data Preparation](#Data Preparation)
1. [Model](#Model)
1. [Training](#Training)
1. [Tricks](#Tricks)
1. [Demo](#Demo)


## Data Preparation

### Recording Data
Becasue the model is trained purely by learning human drive patterns, so the data recorded is critical by:

1) Data needs to be balanced. By natural, the track1 has more left turning than right turning. The initial training will tend to fail easily at right turning because of lack of training data. The solution is to **drive the car in opposite direction**. (make a slow U turn at somewhere wide)

2) Try to record zig-zag pattern. At first, I record by keeping the car driving in the middile of the track as much as possible. But quickly I find it is more useful to teach the model how to recover from edge back to middle.

3) Occational sharp turning is helpful, CNN likes noise!

### Data Preprocess
The raw data recorded is noisy and too large for CNN, the following data processing is implemented:

1) Remove data which were recorded at speed slower than 20mph and acceleration smaller than 0.75. This is because the final test is run at full speed, and there is no brake/speed info feed into DNN.

2) Resize the image by half (from 160x320 to 80x160). It is to save space while not losing too much details.

3) Normalization. It is to speed up model convergence. 

4) Sharpen the image. It helps the CNN to catch the edge, especially at the one unmarked turning.

5) Improve contrast and brightness. It amplifies the details for CNN to catch. In practise, it tends to make the model overfitting , so it might not be useful in this project.

6) Grayscale may not be useful, because the color contrast info is important.

**Note 1: as sample rate is as slow as 10Hz (100ms), image preprocessing should be fast enough in real time.**

**Note 2: In finally submission, 3) to 6) are not used mostly because of overfitting. For example, normalization makes the network converge so fast that only 4 epoches make it overfitting. (I primarily use early stop to overcome overfitting.)**

### Data Augment
It is not used in my implementation, as raw images already acheive very smooth result. But I was planing to try:

1) Horizontal Flip: same effect of reverse driving

2) Motion Blur: useful for less overfitting

3) Rotation/Shift: useful for less overfitting

As it is free to capture as much training data as possible, data augment may not be necessary.

**[Back to top](#table-of-contents)**

## Model
I used the same model as Nvidia published at https://arxiv.org/pdf/1604.07316.pdf. However, there are four notes:

1) **Remove Dropout layers**, which somehow makes the model hard to converge.

2) Tune kernel size a little bit, as the input image size are different.

3) Try with different initilization. The uniform weight initilization makes it fail to converge (50% chance). By using Gaussian standard deviation, the convergence failure decrease to 20%. But I still cannot figure out why it is so easy to fail at convergence.

4) Add a **Tanh activation layer** at the end to limit the output to -1 to 1. (Actually arctan layer is more logically reasonable, but to avoid anomaly, limit to [-1, 1] also makes sense.)


```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to 
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 38, 78, 24)    1824        convolution2d_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 37, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 17, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 15, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 13, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2496)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          2906508     flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================

Total params: 3,159,927
Trainable params: 3,159,927
Non-trainable params: 0
```

## Training
The most dramatic difference in this project is that the training converge very fast, compared to other CNN project. One of the possible reason is the training data is from simulator, which is very clean and less deviation. This indicates three things:

1) It is better to choose adam as optimizer as it remember the previous gradient.

2) Early stopping is necessary, even one more epoch may be a big difference.

3) Tried with learning rate from 0.001 to 0.0001, and at last 0.0001 is used so that slow training would not miss the optima point.

After multiple experiments with the concerns above, I finally choose lr=0.0001, optimizer=adam and loss function is RMSE 


### Overfit
As illustrated above, it is very easy to reach overfitting, so I tried the following methods:

1) Dropout: it doesn't work with my model. It pushes all predictions close to 0.

2) l2 regulation: it works a little bit but not dramatic.

3) early stopping: it is very useful and I take this solution.


**[Back to top](#table-of-contents)**


## Tricks
1) Save the parameters each epoches, and try them on test mode. So that the best fit model can be picked out.

2) As test label (steering angle) is less than one, too many straight driving may lead to network converge to 0. From math point of view, it can be a low gradient result, but it is not useful.

3) The fir_generator is helpful with small DRAM, but I have large enough memory, so I didn't use it. But it should be similar to Tensorflow Image Generator.

4) Observe the test mode in the simulator. For example, if the car drives close to the edge but it doesn't come back to center automatically, it is better to train with more zig-zag data mentioned in chapter 1. For another example, if the car makes a too slow turning at some sharp conditions, supply more sharp turning cases in training dataset.

**[Back to top](#table-of-contents)**


## Demo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=7nb1KC9DAKU
" target="_blank"><img src="http://img.youtube.com/vi/7nb1KC9DAKU/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

**[Back to top](#table-of-contents)**


## License
Nvidia model license

**[Back to top](#table-of-contents)**








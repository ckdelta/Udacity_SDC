# Project 5: Vehicle Detection

In this project, the ultimate goal is to detect the cars in the front anf draw them out, using manually extracted features and machine learning algorithm.


## Table of Contents

1. [Color Space](#Color Space)
1. [HOG Features](#HOG Features)
1. [Color Histogram](#Color Histogram)
1. [Spatially Binned Features](#Spatially Binned Features)
1. [SVM Train and Test](#SVM Train and Test)
1. [Sliding Window Search](#Sliding Window Search)
1. [Heatmap Label](#Heatmap Label)
1. [Tricks & Discussion](#Tricks & Discussion)
1. [Video Demo](#Video Demo)


#Color Space
By default, images are read as RGB. Although it contains color information, but illumination is more important to detect a car on road. Thus I decide to convert to YCbCr before further feature extraction.

The code is implemented in p5_train.py function convert_color.
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/ycrcb.png" alt="YCrCb"/>


#HOG Features
The reason to extract HOG feature is that it represents the edge or shape of the image. To extract the feature better, the following parameters need to be tuned:

1) orientations: it determines how many orientation bins are used; the higher, the better resolution;

2) pixels_per_cell: how many pixels one cell can cover; the lower, the higher resolution but more noise and less generalization;

3) cells_per_block: how many cells to cover the image; the higher, the better resolution but still more noise;

4) HOG channel: using all color channel will represent the features better

Note that, it is not always good to make the HOG to be high resolution, because it loses general features. 

During tuning, the goal is to achieve as high test accuracy as possible. After multiple trials, the following values are used:

```python
orient = 11  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```

The code is implemented in p5_train.py function get_hog_features. The vehicle vs non-vehicle HOG comparison is:
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/HOG_both.png" alt="HOG"/>


#Color Histogram
Color Histogram shows how color is spread in the color space. There is one parameter to tune: the number of bins used to extract color features. I tuned a lot from 16 to 64, but it doesn't affect test accuracy a lot. Although I choose to use 50 at the end, but honestly it is not a effective feature in this case. 

The code is implemented in p5_train.py function color_hist.

#Spatially Binned Features
Because the exact template matching cannot be used directly, so it is necessary to resize the feature image to a smaller size (make it blur) so that robust improves. One parameter need to be tuned: the size of resize. I tried both 64, 32 and 16 and found 16 makes it most robust. But the side effect is false positive is higher.

The code is implemented in p5_train.py function bin_spatial.

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/spatial_both.png" alt="Space"/>


#SVM Train and Test
The vehicle and non-vehicle images are read in sequentially and features are extracted by funtion single_img_features in p5_train.py. The features are insert to a list for future normorlization, shuffle and spit.

```python
for x in os.listdir("vehicles/"):
    dir_path = os.path.join("vehicles/", x)
    for y in os.listdir(dir_path):
        image_path = os.path.join(dir_path, y)
        img = mpimg.imread(image_path)
        feature=single_img_features(img, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
        car_features.append(feature)
```

After the dataset is randomly split to train and test set, I use linear svc to train the model. The reason to use svc is that svc is to optimize for least error, which match the detection purpose. At the end, I acheive 0.9901 accuracy. 

```
Using: 11 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 7386
3.83 Seconds to train SVC...
Test Accuracy of SVC =  0.9901
```

Finally, all parameters are saved in pickle to be used to real time processing. 

#Sliding Window Search
In the begining, I was planning to use multiple size windows to detect cars in near or far away. Later I found if heat map is used, large window can be regenerated with small windows nearby. So the problem becomes to find a smallest window size to catch the most faraway car (smallest car). At the end, I chose the scale size as 1.4 and it proves to reach a good detection resolution.

Except for the most critical parameter scale, there are several paramers to set:

1) ystart(400) and ystop(656) set the vertical area where cars can be present, so that false positive on the air are removed.

2) window size: it works with scale together and here I just use same as the HOG size: 8x8. 

3) cell step: as the window is small enough, putting a step to 2 in between can speed up sliding window search.

The find_cars function is defined in p5_video.py. 

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/multi_windows.png" alt="Multi-window"/>

#Heatmap Label
After multi-window detection has been done, a single car is covered by multiple window. If one window is viewed as a heat point, then by accumulating all window can find an uniform output. It is implemented in p5_video.py with function add_heat, apply_threshold, and draw_labeled_bboxes.

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/window_fusion.png" alt="Heatmap"/>

#Tricks & Discussion

1) This project relies on multiple trail and error process, because manual feature extraction is needed. I would like to try with deep learning aproach, such as RCNN for car detection. I believe some of the processing is not neseccary as deep learning pick up feafures by itself.

2) There is a trade off between false positive and detection coverage. In real application, I think it is necessary to give a confidence factor for each prediection. If it's not confident enough, another sensor or algoorithm needs to act as backup.

3) I tried to put all pipeline together with project 4, it looks good except the processing speed becomes much lower, but still good enough to process video at >30Hz. The combination code is at p4_plus.py.

#Video Demo
My trial error process can be found at P5_progress.ipynb. Traing process is at p5_train.py and video processing is at p5_video.py. Finally, the merge with project4 can be found at p4_plus.py.

Video output is at:

https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/p5_test.mp4

https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/p4_plus_p5.mp4

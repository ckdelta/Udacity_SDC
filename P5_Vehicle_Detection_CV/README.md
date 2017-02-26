# Project4: Vehicle Detection

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
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P5_Vehicle_Detection_CV/output_images/ycrcb.png" alt="YCrCb"/>


#HOG Features
The reason to extract HOG feature is that it represents the edge or shape of the image. To extract the feature better, the following parameters need to be tuned:

1) orientations: it determines how many orientation bins are used; the higher, the better resolution
2) pixels_per_cell: how many pixels one cell can cover; the lower, the higher resolution but more noise and less generalization;
3) cells_per_block: how many cells to cover the image; the higher, the better resolution but still more noise
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


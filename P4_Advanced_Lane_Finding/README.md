# Project4: Advanced Lane Finding

In this project, the ultimate goal is to locate drivable road within the current lane, using traditional computer vision technology, including: camera calibration, image undistortion, perspective transformation, gradient and color threshold, curve fitting etc.


## Table of Contents

1. [Camera Calibration](#Camera Calibration)
1. [Image Undistortion](#Image Undistortion)
1. [Perspective Transformation](#Perspective Transformation)
1. [Gradient & Color Threshold](#Gradient & Color Threshold)
1. [Curve Fitting](#Curve Fitting)
1. [Tricks & Discussion](#Tricks & Discussion)
1. [Video Demo](#Video Demo)


#Camera Calibration
The process can be found in p4_progress.ipynb 2nd cell, at def cam_cal(). After finding all corners by cv2.findChessboardCorners(gray, (9,6), None), points on image plane and world space are found. Then, cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None) is used to get camera calibration matrix. It is saved later with perspective transformation matrix as pickle.

The following is calibration images before and after the process:

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/camera_cal/calibration2.jpg" alt="Before" title="Before" width="256" height="144"/>
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/cal_out.png" alt="After" title="After" width="256" height="144"/>



#Image Undistortion
After camera matrix is ready, it can be used to undistord camera image by cv2.undistort(img, mtx, dist, None, mtx). Because when deriving perspective transformation matrix, stright lane line is needed, so here I will undistord a straight line image as example:

The following is before and after undistortion:

<img src="https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/test_images/straight_lines1.jpg" alt="Before" title="Before" width="256" height="144"/>
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/straight_lines1_undist.jpg" alt="After" title="After" width="256" height="144"/>


#Perspective Transformation
After undistrted straigh line image is ready, next is to derive perspective transformation matrix. I choose two points on the lane lines closet to bottom and two as far away as possible (accurately capture the pixle point by enlarging the image). Then transform it to 1280x720 spaces. Points I choose are:

```python
src=np.float32([[246,690], [1056,690], [694,456], [588,456]])
dst=np.float32([[300,700], [900,700], [900,100],[300,100]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/perspective_trans.png" alt="Transformation" title="Transformation" width="256" height="144"/>

Then, together with camera caliberation data, all can be saved as pickle:

```python
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
dist_pickle["m"] = M
dist_pickle["minv"] = Minv
pickle.dump( dist_pickle, open( "camera_cal/dist_pickle.p", "wb" ) )
```

#Gradient & Color Threshold
Explain in advance, the reason I did perspective transformation first then threshoding is: I only care about the area within lanes, so I focus to tune lane bird-eye binary image clean, rather than tune the whole image area.

By looking at the 6 test images, it is easy to find that test5 is the worst case, as there are reflection and cars as a disturb. So I will show test5 as example but it applies to all other images.

First of all, the reflection is not as bright as lane line, so I plan to use s or h channel threshold to remove the reflection. By tuning a lot, I find h channel offers best result, and here I chose h_thresh=(21, 100). 

However, the h channel thresholding also removes a lot of weak lane lines (especailly right lane), so I plan to add gradient & direction threshold to add more detail information. After tuning a lot, I chose sx_thresh=(40, 100), dir_thresh=(0.2,1.2).

The filtered image looks pretty clean now. Again, I only focus on areas within lanes.

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/binary_lane.png" alt="Threshold" title="Threshold" width="256" height="144"/>

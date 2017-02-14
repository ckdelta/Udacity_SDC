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

<figure><img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/camera_cal/calibration2.jpg" alt="Before" width="256" height="144"><figcaption>Before</figcaption></figure>


#

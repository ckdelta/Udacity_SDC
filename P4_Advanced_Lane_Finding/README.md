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

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/perspective_trans.png" alt="Transformation" title="Transformation" width="512" height="288"/>

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

The filtered image looks pretty clean now. Again, I only focus on areas within lanes. (code is at ipnb [213])

<img src="https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/test_images/test5.jpg" alt="Original" title="Original" width="256" height="144"/>
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/binary_lane.png" alt="Threshold" title="Threshold" width="256" height="144"/>

#Curve Fitting
##Polynomial Function
If curve fitting is applied on binary image directly, noise is too much. So, histogram is used to locate areas in interst first: histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0). Then it is easy to find those points in aggregation and push them into left/right and x/y lists for polynomial fitting: np.polyfit(lefty, leftx, 2)

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/polynomial.png" alt="fit" title="fit" width="512" height="288"/>

##Curve diameter
Once polunomial function coefficients are ready, it is very easy to get diameter by: 

```python
((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
```

The diamters for the image above are: 1769.69167767 1508.82710279(581.109384597 m 495.366464613 m in world space). Quite close to each other.

##Draw back to original image
After apply the line back the bird-eye image, it is easy to re-transform to original perspective, as inverted trasform matrix Minv is ready:

```python
cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
```
It looks good, even though there are tree reflection as a disturb to the pipeline.

<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P4_Advanced_Lane_Finding/output_images/final.png" alt="result" title="result" width="512" height="288"/>

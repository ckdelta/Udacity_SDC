import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pickle
from moviepy.editor import VideoFileClip

class Line():

    def __init__(self):
        #Recent n coefficients queue
        self.recent = []

def camcal():
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for c in os.listdir("camera_cal/"):
        image_path = os.path.join("camera_cal/", c)
        img = mpimg.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    #Image size
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/dist_pickle.p", "wb" ) )

    return mtx, dist, img_size

def threshold(img, sobel_kernel, s_thresh, h_thresh, sx_thresh, dir_thresh):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    h_channel = hsv[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    #Direction Threshold
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)
    direction=np.arctan2(abs_sobely, abs_sobelx)
    sdbinary = np.zeros_like(direction)
    sdbinary[(direction >= dir_thresh[0]) & (direction <= dir_thresh[1])] = 1

    # Threshold s channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold h channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    # Stack each channel
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(h_binary == 1) | ((sxbinary == 1) & (sdbinary == 1)) ] = 1
    return combined_binary

def rad(yvals, x_l, x_r):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(yvals)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    fit_cr_l = np.polyfit(yvals*ym_per_pix, x_l*xm_per_pix, 2)
    fit_cr_r = np.polyfit(yvals*ym_per_pix, x_r*xm_per_pix, 2)
    radius_l = ((1 + (2*fit_cr_l[0]*y_eval + fit_cr_l[1])**2)**1.5)/np.absolute(2*fit_cr_l[0])
    radius_r = ((1 + (2*fit_cr_r[0]*y_eval + fit_cr_r[1])**2)**1.5)/np.absolute(2*fit_cr_r[0])
    return radius_l, radius_r


def pipeline(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undist, M, (1280,720), flags=cv2.INTER_LINEAR)
    binary_warped=threshold(warped,sobel_kernel=7, s_thresh=(180, 190), h_thresh=(21, 100), sx_thresh=(40, 100), dir_thresh=(0.2,1.2))
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_cal = np.polyfit(lefty, leftx, 2)
    right_cal = np.polyfit(righty, rightx, 2)

    #Rule out wrong data
    left_init = left_cal[0]*720*720 + left_cal[1]*720 + left_cal[2]
    right_init = right_cal[0]*720*720 + right_cal[1]*720 + right_cal[2]
    if len(left.recent)>0:
        left_fit=np.mean(left.recent, axis=0)
    else:
        left_fit=[0,0,0]
    #print(left_fit, left_cal)
    if left_init < 210 or left_init > 390 or right_init > 1000 or right_init < 800:
        print("bad luck")
    else:
        if left_fit[2]-left_cal[2]>75:
            print("out lane")
            left_cal=left_fit
        #Moving Average
        left.recent.append(left_cal)
        right.recent.append(right_cal)
        if len(left.recent) > 5:
            del left.recent[0]
            del right.recent[0]
    #Update
    left_fit=np.mean(left.recent, axis=0)
    right_fit=np.mean(right.recent, axis=0)



    #Regulated plot
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #Radius calculation
    xm_per_pix = 3.7/700
    r_l, r_r = rad(ploty, left_fitx, right_fitx)

    #Position calculation
    position_real=(right_init+left_init) * xm_per_pix / 2
    position_ideal = 1280 * xm_per_pix / 2
    distance_from_center=abs(position_ideal - position_real)
    #print(position_ideal, distance_from_center)
    text = 'Left curve: {0:.0f}m, Right curve: {1:.0f}m, Center offset: {2:.0f}cm'.format(r_l, r_r, distance_from_center*100)
    #print(text)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    cv2.putText(result, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return result


if __name__ == '__main__':
    #Run once only
    if False:
        #Camera calibration:
        mtx, dist, img_size=camcal()
        #perspective Transform:
        src=np.float32([[246,690], [1056,690], [694,456], [588,456]])
        dst=np.float32([[300,700], [900,700], [900,100],[300,100]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        dist_pickle["m"] = M
        dist_pickle["minv"] = Minv
        pickle.dump( dist_pickle, open( "./camera_cal/dist_pickle.p", "wb" ) )
    else:
        with open("./camera_cal/dist_pickle.p", mode='rb') as f:
            dist_pickle = pickle.load(f)
            mtx = dist_pickle["mtx"]
            dist = dist_pickle["dist"]
            M = dist_pickle["m"]
            Minv = dist_pickle["minv"]

    #Go through Pipeline
    left = Line()
    right = Line()
    f_out = 'p4.mp4'
    clip = VideoFileClip("project_video.mp4")
    process_clip = clip.fl_image(pipeline)
    process_clip.write_videofile(f_out, audio=False)

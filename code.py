# -*- coding: utf-8 -*-
"""
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""
#%% cell 0: importers
import numpy as np
import cv2
import glob
import matplotlib
#print("matplotlib backend: ",matplotlib.get_backend())
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
import os

#%% cell 1: helper functions
def get_mtx_dist():
    """
    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return (mtx,dist)


def do_undistort(img, mtx, dist, showGUI=False):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    if showGUI:
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
    return dst



def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_color_combo_binary(img,sx_thresh=(20, 100), s_thresh=(170, 255),  showGUI=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Convert to HLS color space and separate the L,S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    img_height, img_length =combined_binary.shape
    vertices = np.array([[(0.1*img_length,img_height),(0.40*img_length, 0.6*img_height), (0.60*img_length, 0.6*img_height), (0.9*img_length,img_height)]], dtype=np.int32)
    combined_binary_masked=region_of_interest(combined_binary, vertices)

    if showGUI:
        # Plot the result
        f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)

        ax2.imshow(color_binary)
        ax2.set_title('color_binary', fontsize=30)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)

        ax3.imshow(combined_binary,cmap='gray')
        ax3.set_title('combined_binary', fontsize=30)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)

        ax4.imshow(combined_binary_masked,cmap='gray')
        ax4.set_title('combined_binary_masked', fontsize=30)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
    return (color_binary,combined_binary, combined_binary_masked)


def get_persp_trans_mtx(img_undist, showGUI=False):
    img_size= (img_undist.shape[1],img_undist.shape[0])
    src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 40, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    if showGUI:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        img=img_undist.copy()
        cv2.polylines(img, np.int_([src]), True, (255,0, 0), thickness=8, lineType=8, shift=0)
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        mpimg.imsave('output_images/straight_lines1_before_perspectiveTf.jpg',img)

        img=img_undist.copy()
        warped_img = cv2.warpPerspective(img, M, (img_undist.shape[1], img_undist.shape[0]))
        cv2.polylines(warped_img, np.int_( [dst]), True, (255,0, 0), thickness=8, lineType=8, shift=0)
        ax2.imshow(warped_img)
        ax2.set_title('Warped Image', fontsize=30)
        mpimg.imsave('output_images/straight_lines1_after_perspectiveTf.jpg',warped_img)
    return (M,Minv)

def do_persp_tf(img, M, showGUI=False):
    """
    Apply a perspective transform to rectify binary image ("birds-eye view").
    """
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    if showGUI:
        # Visualize perspective transform
        if len(img.shape) > 2:
            color_map=None
        else:
            color_map='gray'
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img,cmap=color_map)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(warped,cmap=color_map)
        ax2.set_title('Warped Image', fontsize=30)
    return warped


def find_left_right_line_points(warped_binary):
    # Assuming you have created a warped binary image called "warped_binary"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_binary[warped_binary.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_binary.shape[0] - (window+1)*window_height
        win_y_high = warped_binary.shape[0] - window*window_height
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

    return (leftx,lefty,rightx,righty)



def do_poly_fit(leftx,lefty,rightx,righty, warped_binary=None):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if warped_binary!=None:
        #visualization
        # Generate x and y values for plotting
        fity = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
        fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
        fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

        plt.figure()
        plt.imshow(warped_binary,cmap='gray')
        plt.title('warped_binary and fitted lines')
        plt.plot(fit_leftx, fity, color='yellow')
        plt.plot(fit_rightx, fity, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        straight_lines1_combined_binary_masked_warped_fitted=cv2.cvtColor(255*warped_binary.copy(), cv2.COLOR_GRAY2BGR)
        pts_left = np.int_(np.transpose(np.vstack([fit_leftx, fity])))
        pts_right = np.int_(np.flipud(np.transpose(np.vstack([fit_rightx, fity]))))
        cv2.polylines(straight_lines1_combined_binary_masked_warped_fitted,
                      [pts_left, pts_right], False, (0,255, 255), thickness=4, lineType=8, shift=0)

        cv2.imwrite('output_images/straight_lines1_combined_binary_masked_warped_fitted.jpg',
                     straight_lines1_combined_binary_masked_warped_fitted)


    return (left_fit,right_fit)


def find_left_right_line_points_again(warped_binary, left_fit, right_fit):
    #Now you know where the lines are you have a fit! In the next frame of video you don't need
    #to do a blind search again, but instead you can just search in a margin around the previous
    #line position like this:

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "warped_binary")
    # It's now much easier to find line pixels!
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return (leftx,lefty,rightx,righty)


def calculate_curvature_offset(warped_binary,left_fit,right_fit):
    """
    Determine the curvature of the lane and vehicle position with respect to center.
    Generate some fake data to represent lane-line pixels
    """
    # Define y-value where we want radius of curvature
    yvals = np.linspace(0, warped_binary.shape[0], num=warped_binary.shape[0])# to cover same y-range as image
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3/72 # meters per pixel in y dimension
    xm_per_pix = 3.7/640 # meters per pixel in x dimension

    fit_leftx = left_fit[0]*yvals[-1]**2 + left_fit[1]*yvals[-1] + left_fit[2]
    fit_rightx = right_fit[0]*yvals[-1]**2 + right_fit[1]*yvals[-1] + right_fit[2]

    offset=(0.5*warped_binary.shape[1] - 0.5*(fit_leftx+fit_rightx))*xm_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(yvals*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*yvals[-1]*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*yvals[-1]*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return (left_curverad, right_curverad, offset)


def warp_back(undist,left_fit,right_fit, Minv, print_string,showGUI=False):
    """
    Warp the detected lane boundaries back onto the original image.
    """
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    yvals = np.linspace(0, undist.shape[0]-1, num=undist.shape[0])# to cover same y-range as image
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.int_(np.transpose(np.vstack([left_fitx, yvals])))
    pts_right = np.int_(np.flipud(np.transpose(np.vstack([right_fitx, yvals]))))
    pts = np.hstack((np.array([pts_left]), np.array([pts_right])))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, pts, (0,255, 0))
    cv2.polylines(color_warp, [pts_left, pts_right], False, (255,0, 0), thickness=10, lineType=8, shift=0)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    cv2.putText(result, print_string,(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

    if showGUI:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(undist)
        ax1.set_title('Undistorted Image', fontsize=30)
        ax2.imshow(result)
        ax2.set_title('warped back image', fontsize=30)
    return result

def init():
    mtx, dist = get_mtx_dist()
    img = mpimg.imread('camera_cal/calibration1.jpg')
    dst = do_undistort(img, mtx, dist, showGUI=False)
    mpimg.imsave('output_images/undistort_output.jpg',dst)

    img = mpimg.imread('test_images/straight_lines1.jpg') #straight_lines1
    img_undist = do_undistort(img, mtx, dist, showGUI=False)
    M,Minv=get_persp_trans_mtx(img_undist, showGUI=True)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["M"] = M
    dist_pickle["Minv"] = Minv
    pickle.dump( dist_pickle, open( "./trans_info_pickle.p", "wb" ) )

    return (mtx, dist,M,Minv)

def image_process_pipeline(mtx, dist,M,Minv):
    fit_info=[]
    def process(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
        img_undist = do_undistort(image, mtx, dist, showGUI=False)
        img_undist= gaussian_blur(img_undist, 5)
        color_binary, combined_binary, combined_binary_masked=get_color_combo_binary(img_undist, sx_thresh=(20, 100),s_thresh=(150, 255),  showGUI=False)
        warped_binary =do_persp_tf(combined_binary_masked,M, showGUI=False)

        if not fit_info:
            leftx,lefty,rightx,righty=find_left_right_line_points(warped_binary)
            left_fit,right_fit=do_poly_fit(leftx,lefty,rightx,righty)
        else:
            leftx,lefty,rightx,righty=find_left_right_line_points_again(warped_binary, fit_info[-1][0], fit_info[-1][1])
            left_fit,right_fit=do_poly_fit(leftx,lefty,rightx,righty)
#            weights=[1,1,1,1,1]
            weights=[0.4,0.5,0.6,0.8,0.9]
            for idx,fit in enumerate(fit_info):
                left_fit +=weights[idx]*fit[0]
                right_fit +=weights[idx]*fit[1]

            left_fit /= (1+sum(weights[0:len(fit_info)]))
            right_fit /= (1+sum(weights[0:len(fit_info)]))

        fit_info.append((left_fit,right_fit))
        if len(fit_info)>5:
            del fit_info[0]
        assert len(fit_info)<=5

        left_curverad, right_curverad, offset=calculate_curvature_offset(warped_binary,left_fit,right_fit)
        print_string = "curverad in meters: %.2f m, offset: %.3f m." % (0.5*(left_curverad+right_curverad), offset)
        result=warp_back(img_undist,left_fit,right_fit, Minv, print_string, showGUI=False)
        return result

    return process
#%% cell 2: initialization to get camera matrix, distortion coefficients, pesperctive transformation matrix and its inverse

if not os.path.exists("./trans_info_pickle.p"):
    mtx, dist,M,Minv=init()

with open("./trans_info_pickle.p", mode='rb') as f:
    dist_info = pickle.load(f)
    mtx = dist_info["mtx"]
    dist = dist_info["dist"]
    M = dist_info["M"]
    Minv = dist_info["Minv"]

#%% cell 3: details of the use of the image precessing pipeline to a single image

img = mpimg.imread('test_images/straight_lines1.jpg') #straight_lines1
img_undist = do_undistort(img, mtx, dist, showGUI=False)
mpimg.imsave('output_images/straight_lines1_undistorted.jpg',img_undist)
img_undist= gaussian_blur(img_undist, 5)
warped_img =do_persp_tf(img_undist,M, showGUI=True)


color_binary, combined_binary, combined_binary_masked=get_color_combo_binary(img_undist, sx_thresh=(20, 100), s_thresh=(150, 255),showGUI=True)
mpimg.imsave('output_images/straight_lines1_color_binary.jpg',color_binary)
mpimg.imsave('output_images/straight_lines1_combined_binary.jpg',combined_binary, cmap='gray')
mpimg.imsave('output_images/straight_lines1_combined_binary_masked.jpg',combined_binary_masked*255, cmap='gray')

warped_binary =do_persp_tf(combined_binary_masked,M, showGUI=False)
mpimg.imsave('output_images/straight_lines1_combined_binary_masked_warped.jpg',warped_binary*255, cmap='gray')

leftx,lefty,rightx,righty=find_left_right_line_points(warped_binary)

left_fit,right_fit=do_poly_fit(leftx,lefty,rightx,righty, warped_binary)

#leftx,lefty,rightx,righty=find_left_right_line_points_again(warped_binary, left_fit, right_fit)
#left_fit,right_fit=do_poly_fit(leftx,lefty,rightx,righty, warped_binary)

left_curverad, right_curverad, offset=calculate_curvature_offset(warped_binary,left_fit,right_fit)
print_string = "curverad in meters: %.2f m, %.2f m,offset: %.3f m." % (left_curverad,right_curverad, offset)
warped_back=warp_back(img_undist,left_fit,right_fit, Minv, print_string, showGUI=True)
mpimg.imsave('output_images/straight_lines1_warped_back.jpg',warped_back)

#%% cell 4: image processing pipeline used to test images

def process(image):
    img_undist = do_undistort(image, mtx, dist, showGUI=False)
    img_undist= gaussian_blur(img_undist, 5)
    color_binary, combined_binary, combined_binary_masked=get_color_combo_binary(img_undist, sx_thresh=(20, 100),s_thresh=(150, 255),  showGUI=False)
    warped_binary =do_persp_tf(combined_binary_masked,M, showGUI=False)
    leftx,lefty,rightx,righty=find_left_right_line_points(warped_binary)
    left_fit,right_fit=do_poly_fit(leftx,lefty,rightx,righty)
    left_curverad, right_curverad, offset=calculate_curvature_offset(warped_binary,left_fit,right_fit)
    print_string = "curverad in meters: %.2f m, %.2f m, offset: %.3f m." % (left_curverad, right_curverad, offset)
    result=warp_back(img_undist,left_fit,right_fit, Minv, print_string, showGUI=False)
    return result

images = glob.glob('test_images/*.jpg')
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    mpimg.imsave(fname.replace('test_images','output_images'),process(img))

#%% cell 5: video processing to dectect lanes
video_output = 'output_images/video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
pipeline=image_process_pipeline(mtx, dist,M,Minv)
video_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False)


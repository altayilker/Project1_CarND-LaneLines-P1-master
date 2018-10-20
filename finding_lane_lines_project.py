# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:25:36 2018

@author: u19l65
"""

# Do all the relevant imports
# Do relevant imports

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    return lines #line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


########################################################################################################################


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(image)  
ax.set(title='Real picture')

gray = grayscale(image)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(gray, cmap='gray') # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
ax.set(title='Gray picture')


# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
masked_edges = canny(blur_gray, low_threshold, high_threshold)


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 20
max_line_gap = 20
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
# Iterate over the output "lines" and draw lines on the blank
draw_lines(line_image, lines, color=[255, 0, 0], thickness=10)
# Create a "color" binary image to combine with line image
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(color_edges)
ax.set(title='color_edges')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(line_image)
ax.set(title='line_image')

# Draw the lines on the edge image
combo = weighted_img(color_edges, line_image, α=0.8, β=1., γ=0.) 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(combo)
ax.set(title='Hough transform')


# Draw the lines on the real image
combo2 = cv2.addWeighted(image, 0.8, line_image, 1, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(combo2)
ax.set(title='Hough transform')


# Mask the image
vertices =  np.array( [[[100, 539],[900, 539],[520, 320],[440, 320]]] )
masked_image = region_of_interest(image, vertices)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(masked_image )
ax.set(title='masked_image ')

# Draw the lines on the real image
combo3 = cv2.addWeighted(masked_image, 0.8, line_image, 1, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(combo3 )
ax.set(title='combo3 ')


# Draw the lines on the real image
NEW_AREA = masked_image&line_image
plt.figure() 
plt.imshow(NEW_AREA)
combo4 = cv2.addWeighted(image, 0.8, NEW_AREA, 1, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(combo4 )
ax.set(title='masked_image&line_image ')




######################################## Video part ################################

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

     gray = grayscale(image)
 
     # Define a kernel size and apply Gaussian smoothing
     kernel_size = 5
     blur_gray = gaussian_blur(gray, kernel_size)
 
     # Define our parameters for Canny and apply
     low_threshold = 50
     high_threshold = 150
     masked_edges = canny(blur_gray, low_threshold, high_threshold)
  
     # Define the Hough transform parameters
     # Make a blank the same size as our image to draw on
     rho = 2
     theta = np.pi/180
     threshold = 15# =============================================================================
     min_line_length = 20
     max_line_gap = 20
     line_image = np.copy(image)*0 #creating a blank to draw lines on
 
     # Run Hough on edge detected image
     lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
     # Iterate over the output "lines" and draw lines on the blank
     draw_lines(line_image, lines, color=[255, 0, 0], thickness=10)
 
     # Mask the image
     vertices =  np.array( [[[100, 539],[900, 539],[520, 320],[440, 320]]] )
     masked_image = region_of_interest(image, vertices)
 
     # Draw the lines on the real image
     NEW_AREA = masked_image&line_image
 
     combo4 = cv2.addWeighted(image, 0.8, NEW_AREA, 1, 0)  

     return combo4


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)






## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_undist.jpg "Undistorted Calibration Image"
[image2]: ./output_images/undist.jpg "Road Undistorted"
[image3]: ./output_images/binary.png "Binary"
[image4]: ./output_images/birds-eye_view.png "Warp"
[image5]: ./output_images/birds-eye_view_binary.png "Warp Binary"
[image6]: ./output_images/color_fit_lines.png "Fit Visual"
[image7]: ./output_images/result.png "Output"
[video1]: ./output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the lines 77-118 of the file 'Solution.py'.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 122 through 214 in 'Solution.py').  Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called 'getWarped()', which appears in lines 217 through 226 in the file 'Solution.py'.  This function takes as inputs an image (`img`).  I chose the hardcode the source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 250, 0        | 
| 690, 450      | 1060, 0       |
| 1060, 690     | 1060, 690     |
| 250, 690      | 250, 690      |

The result of perspective transformation and the corresponding binary are shown below:

![alt text][image4]
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I found start points of the lines in function 'getLaneStartX' (lines 229-250 of 'Solution.py') and fit my lane lines with a 2nd order polynomial in function 'getPolynomials' (lines 252-317 of 'Solution.py'):

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this as it was described in the lectures (lines 371 through 383 of 'Solution.py')

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 390 through 406 in my code in 'Solution.py' in the function 'getOutputImage'.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a ![video](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First, I tried several color channels (RGB, HLS and grayscale), image gradients and various combinations thereof to get a binary image. I found that S channel together with gradients worked best for me on test images. However, I also noticed that in some cases adding thresholded gradient introduces noise which hardens identifying the lines correctly. Unfortunately, I could not bypass using gradient because otherwise on some frames the pipeline failed to find lines on S channel basis at all.

Another concern of mine is weather conditions. My pipeline was tested on pictures with a high quality road and sunny weather. I doubt that the channels and their threshold values that I chose would also perform well in other circumstances. I believe that my pipeline requires tuning to make in more generalized.
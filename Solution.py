import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

class LaneFinder:
    path_to_mtx = 'mtx.npy'
    path_to_dist = 'dist.npy'
    left_fit = None
    right_fit = None
    out_img_folder = 'output_images/'

    def __init__(self, calibrate_anew=False):
        if not calibrate_anew:
            try:
                self.mtx = np.load(self.path_to_mtx)
                self.dist = np.load(self.path_to_dist)
                return
            except IOError:
                pass
        self.calibrate()

    def processImage(self, img, quiet=True):
        self.quiet = quiet
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        if not self.quiet:
            cv2.imwrite(self.out_img_folder + 'undist.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        grad_combined = self.getGradientImgCombined(undist)

        color_combined = self.getColorCombined(undist, gray_thresh=(190, 255), r_thresh=(210, 255))

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(grad_combined)
        combined_binary[(grad_combined == 1) | (color_combined == 1)] = 1
        if not self.quiet:
            mpimg.imsave(self.out_img_folder + 'binary.png', combined_binary, cmap='gray')

        self.getWarped(combined_binary)
        if not self.quiet:
            img_size = (undist.shape[1], undist.shape[0])
            warped = cv2.warpPerspective(undist, self.M, img_size, flags=cv2.INTER_LINEAR)
            mpimg.imsave(self.out_img_folder + 'birds-eye_view.png', warped, cmap='gray')
            mpimg.imsave(self.out_img_folder + 'birds-eye_view_binary.png', self.binary_warped, cmap='gray')

        leftx_base, rightx_base = self.getLaneStartX()

        left_fit, right_fit = self.getPolynomials(leftx_base, rightx_base)

        self.getPlottingValues(left_fit, right_fit)

        if not self.quiet:
            self.visualizeLineSearch()

        left_curverad, right_curverad = self.getCurvatureMeters()
        # if not self.quiet:
            # print(left_curverad, 'm', right_curverad, 'm')
        curv = (left_curverad + right_curverad) / 2
        center_offset = self.getCenterOffsetMeters(img)
        # print (center_offset)

        out_img = self.getOutputImage(img, undist)
        font = cv2.FONT_HERSHEY_SIMPLEX
        out_img = cv2.putText(out_img, "Curvature: " + str(int(curv)) + 'm',(20,40), font, 1, (1.0,1.0,1.0), 2, cv2.LINE_AA)
        if center_offset < 0:
            pos = 'right'
        else:
            pos = 'left'
        out_img = cv2.putText(out_img, str(abs(round(center_offset, 2))) + 'm to the ' + pos + ' of the center',(20,80), font, 1, (1.0,1.0,1.0), 2, cv2.LINE_AA)
        # plt.imshow(result)
        if not self.quiet:
            mpimg.imsave(self.out_img_folder + 'result.png', out_img)
        return out_img

    def calibrate(self):
        pathToImages = 'camera_cal/'
        calFiles = [f for f in os.listdir(pathToImages) if os.path.isfile(pathToImages.join(f))]
        images = glob.glob('camera_cal/calibration*.jpg')

        # prepare object points
        nx = 9
        ny = 6

        objpoints = []
        imgpoints = []
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Make a list of calibration images
        for image in images:
            img = mpimg.imread(image)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, draw corners
            if ret == True:
                # if not self.quiet:
                    # Draw and display the corners
    #               cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    #               plt.imshow(img)
    #               plt.show()
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist

        undist = cv2.undistort(cv2.imread(images[0]), self.mtx, self.dist, None, self.mtx)
        cv2.imwrite(self.out_img_folder + 'calibration_undist.jpg', undist)
        np.save(self.path_to_mtx, mtx)
        np.save(self.path_to_dist, dist)



    def getThresholdedSobel(self, undist, orient='x', thresh_min=20, thresh_max=100):
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)

        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        else:
            return None

        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return binary_output


    def mag_thresh(self, undist, sobel_kernel=3, mag_thresh=(100, 200)):
        # Convert to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output


    # Define a function to threshold an image for a given range and Sobel kernel
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output


    def getGradientImgCombined(self, undist):
        ksize = 3

        gradx = self.getThresholdedSobel(undist, orient='x')
        grady = self.getThresholdedSobel(undist, orient='y')

        magn = self.mag_thresh(undist)
        direction = self.dir_threshold(undist, sobel_kernel=ksize, thresh=(0, np.pi/2))

        combined = np.zeros_like(direction)
        combined[((gradx == 1) & (grady == 1)) | ((magn == 1) & (direction == 1))] = 1
        return combined

    def getColorCombined(self, undist,
                         gray_thresh=(180, 255),
                         r_thresh=(230, 255)):
    #     gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    #     gray_binary = np.zeros_like(gray)
    #     gray_binary[(gray > gray_thresh[0]) & (gray <= gray_thresh[1])] = 1

        R = undist[:,:,0]
        r_binary = np.zeros_like(R)
        r_binary[(R > r_thresh[0]) & (R <= r_thresh[1])] = 1

        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]

        thresh = (90, 255)
        s_binary = np.zeros_like(S)
        s_binary[(S > thresh[0]) & (S <= thresh[1])] = 1

        combined = np.zeros_like(r_binary)
    #     combined[((gray_binary == 1) | (r_binary == 1))] = 1
    #     combined[((r_binary == 1) & (s_binary == 1))] = 1
        combined[((s_binary == 1))] = 1
    #     combined[((r_binary == 1))] = 1

        return combined


    def getWarped(self, img):
        img_size = (img.shape[1], img.shape[0])

        src = np.float32([(590, 450), (690, 450), (1060, 690), (250, 690)])
        dst = np.float32([(250, 0), (1060, 0), (1060, 690), (250, 690)])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        self.binary_warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)


    def getLaneStartX(self):
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # if not self.quiet:
        #     plt.plot(histogram)
        #     plt.show()

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)

        left_half = histogram[:midpoint]
        right_half = histogram[midpoint:]

        count = 50

        l_tmp = left_half.argsort()[-count:][::-1]
        r_tmp = right_half.argsort()[-count:][::-1] + midpoint

        leftx_base = np.average(l_tmp)
        rightx_base = np.average(r_tmp)

        return leftx_base, rightx_base

    def getPolynomials(self, leftx_base, rightx_base):
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(self.binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 100
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(self.lefty, self.leftx, 2)
        right_fit = np.polyfit(self.righty, self.rightx, 2)

        # print(left_fit)
        # print(right_fit)
        # print()

        self.left_fit = self.sanity_check(left_fit, self.left_fit)
        self.right_fit = self.sanity_check(right_fit, self.right_fit)

        # if not self.quiet:
            # plt.imshow(out_img)
            # plt.show()
        return self.left_fit, self.right_fit

    def sanity_check(self, new, old):
        if old == None:
            old = new
            return old

        deltas = [100, 100, 10]
        for i in range(len(deltas)):
            big = max(abs(new[i]), abs(old[i]))
            small = min(abs(new[i]), abs(old[i]))
            if big / small > deltas[i]:
                # print('SANITY: i=' + str(i) + ', old=' + str(old) + ', new=' + str(new))
                # print()
                return old
        old = new
        return old

    def getPlottingValues(self, left_fit, right_fit):
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        self.left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        self.right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]

    def visualizeLineSearch(self):
        # Create an image to draw on and an image to show the selection window
        out_img = (np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255).astype(np.uint8)
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-self.margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+self.margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-self.margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+self.margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        mpimg.imsave(self.out_img_folder + 'color_fit_lines.png', result)
        # plt.imshow(result)
        # plt.plot(self.left_fitx, self.ploty, color='yellow')
        # plt.plot(self.right_fitx, self.ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()

    def getCurvatureMeters(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        y_eval = np.max(self.ploty)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.lefty*ym_per_pix, self.leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty*ym_per_pix, self.rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        return left_curverad, right_curverad

    def getCenterOffsetMeters(self, img):
        xm_per_pix = 3.7/700
        lane_center = (self.left_fitx[-1] + self.right_fitx[-1])/2
        return ((lane_center - img.shape[1] / 2) * xm_per_pix)

    def getOutputImage(self, img, undist):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))

        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


laneFinder = LaneFinder()
img = mpimg.imread('test_images/test3.jpg')
result = laneFinder.processImage(img, quiet=False)

plt.imshow(result)
plt.show()


# from moviepy.editor import VideoFileClip
# video = VideoFileClip('project_video.mp4')
# processed_video = video.fl_image(laneFinder.processImage)
# processed_video.write_videofile('output.mp4', audio=False)
## Writeup

**Advanced Lane Finding Project**

The check list of the goals of this project are the following:

- [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- [x] Apply a distortion correction to raw images.
- [x] Use color transforms, gradients, etc., to create a thresholded binary image.
- [x] Apply a perspective transform to rectify binary image ("birds-eye view").
- [x] Detect lane pixels and fit to find the lane boundary.
- [x] Determine the curvature of the lane and vehicle position with respect to center.
- [x] Warp the detected lane boundaries back onto the original image.
- [x] Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

> Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Camera Calibration

The function is implemented in `python/camera_calibration.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at `z=0`, such that the object points are the same for each calibration image. The object points are generated in `_generate_object_points` and it's a 3D array has the same length of total number of images.

Then I load the images using `load_images` given `image_path` to read all of the images for camera calibration and run `detect_chess_board_corners` on them to get the images points. The return array is a 3D array and containing a list of image points (coming from all images).

Then the final step is to feed the obejct points and images points into `cv2.calibrateCamera` to get the final camera matrix `KK` and distortion coefficents `Kc`.

After I got the camera KK and Kc, I implemented a function `undistort_images` to undistort all images and return a list of undistorted images.

![Example of camera calibration result](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/camera_calibration_example.jpg)

I save the camera KK and Kc in a file called `camera.json` and will use it the actual lane detection pipeline.

Camera matrix json
```json
{
 "KK": [
  [
   484.3902884693779,
   0.0,
   359.50000343668137
  ],
  [
   0.0,
   730.5665190852465,
   639.4999981458845
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "Kc": [
  [
   -0.04327892605564115,
   0.00008322703351661759,
   -0.008936257678709005,
   0.024703269877233883,
   2.5638661810637675e-7
  ]
 ]
}
```

![All images used for calibration with detected pattern corners drawn](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/undistort_images_with_detected_corners.jpg)

### Pipeline

All related code of the lane detection pipeline is located in `python/lane_detection.py`

#### 1. Provide an example of a distortion-corrected image.

Here is an undistorted image using camera KK and Kc in `camera.json`.

![Example of image undistortion result](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/camera_calibration_example_2.jpg)

#### 2. Image warping

Image warping is implemented in function `preprocess_image`. Source ROI corners are defined as `ROI_CORNERS` and dest (warped) ROI corners are defined as `WARPED_ROI_CORNERS`.

```py
IMAGE_SHAPE = (720, 1280)
ROI_CORNERS = np.float32([(575, 464),
                          (707, 464),
                          (258, 682),
                          (1049, 682)])
WARPED_ROI_CORNERS = np.float32([(450, 0),
                                 (IMAGE_SHAPE[1] - 450, 0),
                                 (450, IMAGE_SHAPE[0]),
                                 (IMAGE_SHAPE[1] - 450, IMAGE_SHAPE[0])])
```

Homography for perspective transform is fone by using `cv2.getPerspectiveTransform`.

![Warped images](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/straight_lines1_edge_images.jpg)

#### 3. Binary image creation
The purpose of this step is to find the proper binary image for later processing. and the input image is already warped.

This step is implemented in function `preprocess_image`.

The first step is to examine the color spaces to see if we can find a channel of a particular color space where lane line is visually outstanding.

![Example of color spaces](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/straight_lines1_color_transform_comparison.jpg)

Depending on the obsevation in the color channels listed above, the processing logic is defined as

```py
# Use H and S channel in HSL image since the lane color is more out-standing than others
s_thres = 125
h_thres = 50
l_thres = 210
b_thres = 125

hls_h_mask = hls_image[:, :, 0] < h_thres
hls_l_mask = hls_image[:, :, 1] > l_thres
hls_s_mask = hls_image[:, :, 2] > s_thres
lab_b_mask = lab_image[:, :, 2] > b_thres

lane_image = ((hls_s_mask & lab_b_mask & hls_h_mask) | (hls_l_mask & hls_h_mask)).astype(np.uint8) * 255
```

#### 4. Lane searching and polynomial fitting

It's implemeted in function `search_lane_by_sliding_window` and `search_lane_by_previous_result`. As the name suggests, the first function is used when there is no input of the polynomial coefficients from previous results (mostlikely first frame, or in the previous frame, polynomimal fitting cannot find a good estimation and we have to start over from scratch). The second one will use the previous polynomial and filter out the valid image points in the the ROI and do polynomial again.

![Lane searching by using previous polynomial results](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/project_video.mp4_1_search_lane_using_previous_result.jpg)

![Lane searching by sliding window](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/straight_lines1_lane_searching.jpg)

#### 5. Curvature calculation

It's implemented in function `measure_curvature_pixels`.

#### 6. Final result output

![Example of final result](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/straight_lines1_result.jpg)

### Pipeline (video)

#### 1. Final result outputs

Here's the [project_video](https://github.com/kunlin596/CarND-Data/blob/master/P2-advanced-lane-lines/outputs/project_video.mp4) with lane detection results overlayed.

---

### Discussion

TODO
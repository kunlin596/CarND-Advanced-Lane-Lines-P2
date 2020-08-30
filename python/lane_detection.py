#!/usr/bin/env python3

import cv2
import glob
import os
import numpy as np

import matplotlib.pyplot as plt

from IPython import embed
import matplotlib
import logging

np.set_printoptions(suppress=True, precision=5)
plt.ion()

try:
    import colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s[%(levelname)s] %(name)s: %(message)s'))
    log = colorlog.getLogger('lane_detection')
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
except ImportError as e:
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger("lane_detection")

matplotlib.use('TkAgg')

# Hard coded for test images, these values are tweaked using test_images/straight_lines1.jpg
IMAGE_SHAPE = (720, 1280)
# ROI_CORNERS = np.float32([[IMAGE_SHAPE[1] * 0.5 - 65, IMAGE_SHAPE[0] / 2 + 100],
#                           [IMAGE_SHAPE[1] * 0.2, IMAGE_SHAPE[0] - 50],
#                           [IMAGE_SHAPE[1] * 0.8, IMAGE_SHAPE[0] - 50],
#                           [IMAGE_SHAPE[1] * 0.5 + 65, IMAGE_SHAPE[0] / 2 + 100]])


# WARPED_ROI_CORNERS = np.float32([[IMAGE_SHAPE[1] * 0.2, 0],
#                                  [IMAGE_SHAPE[1] * 0.2, IMAGE_SHAPE[0]],
#                                  [IMAGE_SHAPE[1] * 0.8, IMAGE_SHAPE[0]],
#                                  [IMAGE_SHAPE[1] * 0.8, 0]])


ROI_CORNERS = np.float32([(575, 464),
                          (707, 464),
                          (258, 682),
                          (1049, 682)])
WARPED_ROI_CORNERS = np.float32([(450, 0),
                                 (IMAGE_SHAPE[1] - 450, 0),
                                 (450, IMAGE_SHAPE[0]),
                                 (IMAGE_SHAPE[1] - 450, IMAGE_SHAPE[0])])

Y_METER_PER_PIXEL = 30 / 720
X_METER_PER_PIXEL = 3.7 / 700


def get_homography():
    src = ROI_CORNERS
    dst = WARPED_ROI_CORNERS
    return cv2.getPerspectiveTransform(src, dst)


def load_images(imagePath):
    if not os.path.exists(imagePath):
        raise Exception('%s is not valid.' % (imagePath))

    images = {}
    image_paths = glob.glob(os.path.join(imagePath, '*.jpg'))

    for index, path in enumerate(image_paths):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        log.debug('loaded %s', path)
        image_name = os.path.basename(path).split('.')[0]
        images[image_name] = image
    return images


def search_lane(image_name, image, KK, window_size=None, show=False, output_images=False):
    """Search lane curve in warped image

    Arguments:
        image {[type]} -- [description]
        KK {[type]} -- [description]

    Keyword Arguments:
        window_size {[type]} -- [description] (default: {None})
        show {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    if window_size is None:
        window_size = (125, 40)

    #
    # Find out the starting position of both search windows
    #

    image[:, :350] = 0
    image[:, IMAGE_SHAPE[1] - 350:] = 0

    y, x = np.nonzero(image[image.shape[0] // 2:, :])
    bins = int(image.shape[1] / 10 + 0.5)
    h, edges = np.histogram(x, bins=bins)

    median_index = len(h) // 2
    left_max_bin = np.argmax(h[:median_index])
    left_max = h[:median_index].max()
    right_max_bin = np.argmax(h[median_index:]) + median_index
    right_max = h[median_index:].max()

    # This value should be computed using camera KK and extrinsics,
    # here, it is estimated using straight_line.jpg
    lane_width_in_pixel = 400

    # The distance between the 2 peaks should be roughly `lane_width_in_pixel`,
    # Use the 2nd peak is way small than the 1st peak, use 1st peak to fix the 2nd peak
    left_x = edges[left_max_bin: left_max_bin + 1].mean()
    right_x = edges[right_max_bin: right_max_bin + 1].mean()

    if right_max / left_max < 0.3:
        right_x = left_x + lane_width_in_pixel
    elif left_max / right_max < 0.3:
        left_x = right_x - lane_width_in_pixel

    #
    # Initialize initial left and right search locations
    #
    window_height = window_size[1]
    current_y = image.shape[0] - window_height / 2
    current_left_x = int(left_x + 0.5)
    current_right_x = int(right_x + 0.5)

    def get_window_patch(current_x, current_y, window_size):
        corners = (cv2.boxPoints(((current_x, current_y), window_size, 0)) + 0.5).astype(np.int32)
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
        windows_image = image[min_xy[1]: max_xy[1], min_xy[0]: max_xy[0]]
        return windows_image, min_xy, corners

    if output_images:
        if not show:
            plt.ioff()

        plt.figure(figsize=(30, 20))
        plt.imshow(image, cmap='gray')
        plt.xlim([0, IMAGE_SHAPE[1]])
        plt.ylim([IMAGE_SHAPE[0], 0])
        plt.plot(WARPED_ROI_CORNERS[[0, 1, 3, 2, 0]][:, 0], WARPED_ROI_CORNERS[[0, 1, 3, 2, 0]][:, 1], 'g', linewidth=3)

        if show:
            plt.show(block=False)

    current_window_size = window_size

    # All valid lane pixels in full size image
    left_lane_points = []
    right_lane_points = []

    #
    # Main loop
    #
    while current_y > 0:
        left_patch, top_left_corner_left, left_window_corners = get_window_patch(current_left_x, current_y, current_window_size)
        right_patch, top_left_corner_right, right_window_corners = get_window_patch(current_right_x, current_y, current_window_size)

        # All valid pixels on left lane ROI window region
        valid_left_y, valid_left_x = np.nonzero(left_patch)
        valid_left_window_points = np.vstack([valid_left_x, valid_left_y]).T + top_left_corner_left
        left_window_center = left_window_corners.mean(axis=0)

        # All valid pixels on right lane ROI window region
        valid_right_y, valid_right_x = np.nonzero(right_patch)
        valid_right_window_points = np.vstack([valid_right_x, valid_right_y]).T + top_left_corner_right
        right_window_center = right_window_corners.mean(axis=0)

        is_left_valid = len(valid_left_window_points) > 0
        is_right_valid = len(valid_right_window_points) > 0

        # Use all valid pixels in the window image, if no valid points are found, use window center.
        if not is_left_valid and not is_right_valid:
            valid_left_window_points = [left_window_center]
            valid_right_window_points = [right_window_center]
        elif is_left_valid and not is_right_valid:
            valid_right_window_points = [[left_window_center[0] + lane_width_in_pixel, left_window_center[1]]]
        elif not is_left_valid and is_right_valid:
            valid_left_window_points = [[right_window_center[0] - lane_width_in_pixel, right_window_center[1]]]
        else:
            lane_width_in_pixel = valid_right_window_points[:, 0].mean() - valid_left_window_points[:, 0].mean()

        valid_left_window_points = np.asarray(valid_left_window_points)
        valid_right_window_points = np.asarray(valid_right_window_points)

        current_left_x = valid_left_window_points.mean(axis=0)[0]
        current_right_x = valid_right_window_points.mean(axis=0)[0]

        left_lane_points.extend(valid_left_window_points.tolist())
        right_lane_points.extend(valid_right_window_points.tolist())

        # Move window up
        current_y -= window_size[1]

        if output_images:
            plt.plot(valid_left_window_points[:, 0], valid_left_window_points[:, 1], 'r.')
            plt.plot(valid_right_window_points[:, 0], valid_right_window_points[:, 1], 'b.')
            plt.plot(left_window_center[0], left_window_center[1], 'b.')
            plt.plot(right_window_center[0], right_window_center[1], 'r.')

            # Plot window
            plt.plot(left_window_corners[:, 0][[0, 1, 2, 3, 0]], left_window_corners[:, 1][[0, 1, 2, 3, 0]], 'r')
            plt.plot(right_window_corners[:, 0][[0, 1, 2, 3, 0]], right_window_corners[:, 1][[0, 1, 2, 3, 0]], 'b')

    left_lane_points = np.array(left_lane_points).reshape(-1, 2)
    right_lane_points = np.array(right_lane_points).reshape(-1, 2)

    left_poly = np.polyfit(left_lane_points[:, 1], left_lane_points[:, 0], 2)
    estimated_left_x = np.polyval(left_poly, np.arange(0, image.shape[0]))

    right_poly = np.polyfit(right_lane_points[:, 1], right_lane_points[:, 0], 2)
    estimated_right_x = np.polyval(right_poly, np.arange(0, image.shape[0]))
    log.debug('left_poly=%s, right_poly=%s', left_poly, right_poly)

    if output_images:
        if estimated_left_x is not None:
            plt.plot(estimated_left_x, np.arange(0, image.shape[0]), linewidth=3, color='b')

        if estimated_right_x is not None:
            plt.plot(estimated_right_x, np.arange(0, image.shape[0]), linewidth=3, color='r')

        if not show:
            plt.show(block=False)

        plt.savefig('output_images/%s_lane_searching.jpg' % image_name)
        plt.close()

    return left_poly, right_poly


def measure_curvature_pixels(poly, y_value):
    """Measure lane curvature

    Arguments:
        poly {[type]} -- [description]
        v_value {[type]} -- [description]
    """
    A = poly[0]
    B = poly[1]
    return (1 + (2 * A * y_value * B) ** 2) ** (1.5) / abs(2 * A)


def preprocess_image(image, image_name, show, output_images=False):
    log.debug('current image_name=%s', image_name)

    # Warp image
    homography = cv2.getPerspectiveTransform(ROI_CORNERS, WARPED_ROI_CORNERS)
    warped_image = cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]))

    hls_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HLS)
    hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2LAB)
    luv_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2LUV)

    if output_images:
        if not show:
            plt.ioff()
        plt.figure(figsize=(30, 20))
        plt.suptitle('Channel comparison of difference color spaces', fontsize=32)
        num_color_spaces = 5
        num_channels = 3

        for i in range(num_channels):
            plt.subplot(num_color_spaces, num_channels, i + 1)
            plt.title('rgb channel[%d]' % i)
            plt.imshow(warped_image[:, :, i], cmap='gray')

            plt.subplot(num_color_spaces, num_channels, i + num_channels + 1)
            plt.title('hls channel[%d]' % i)
            plt.imshow(hls_image[:, :, i], cmap='gray')

            plt.subplot(num_color_spaces, num_channels, i + num_channels * 2 + 1)
            plt.imshow(hsv_image[:, :, i], cmap='gray')
            plt.title('hsv channel[%d]' % i)

            plt.subplot(num_color_spaces, num_channels, i + num_channels * 3 + 1)
            plt.imshow(lab_image[:, :, i], cmap='gray')
            plt.title('lab channel[%d]' % i)

            plt.subplot(num_color_spaces, num_channels, i + num_channels * 4 + 1)
            plt.imshow(luv_image[:, :, i], cmap='gray')
            plt.title('luv channel[%d]' % i)

        plt.tight_layout()
        if show:
            plt.show(block=False)
        plt.savefig('output_images/%s_color_transform_comparison.jpg' % image_name)
        plt.close()

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

    # Undistort image
    kernel_size = 3

    # Create sobel image
    min_mag = 25
    max_mag = 255
    # NOTE: Perhaps put a constraint on the direction of the gradient? Maybe too risky when lane is too curved.
    grad_x = np.abs(cv2.Sobel(lane_image, cv2.CV_64F, 1, 0, kernel_size))
    grad_y = np.abs(cv2.Sobel(lane_image, cv2.CV_64F, 0, 1, kernel_size))
    sobel_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    sobel_mag_normalized = (sobel_mag / sobel_mag.max() * 255).astype(np.uint8)
    sobel_mag_mask = ((min_mag < sobel_mag_normalized) & (sobel_mag_normalized < max_mag)).astype(np.uint8) * 255

    # NOTE: Canny vs sole sobel?
    # canny_image = cv2.Canny(lane_image.astype(np.uint8), 155, 255)
    # warped_canny_image = cv2.warpPerspective(canny_image, homography, (canny_image.shape[1], canny_image.shape[0]))
    # warped_image = cv2.warpPerspective(image, homography, (canny_image.shape[1], canny_image.shape[0]))

    if output_images:
        if not show:
            plt.ioff()

        plt.figure(figsize=(30, 20))
        plt.suptitle('Warped edge images', fontsize=32)

        plt.subplot(231)
        plt.title('Original color image with ROI corners')
        image2 = cv2.polylines(image, (ROI_CORNERS[[0, 1, 3, 2]] + 0.5).astype(np.int32).reshape(1, -1, 2), True, (255, 0, 0), 1)
        plt.imshow(image2)

        plt.subplot(232)
        plt.title('Warped image')
        plt.imshow(warped_image)

        plt.subplot(233)
        plt.title('HLS S channel thresholded')
        plt.imshow(lane_image, cmap='gray')
        # plt.imshow(warped_sobel_mask, cmap='gray')

        plt.subplot(234)
        plt.title('Sobel grad x')
        plt.imshow(grad_x, cmap='gray')

        plt.subplot(235)
        plt.title('Sobel grad y')
        plt.imshow(grad_y, cmap='gray')

        plt.subplot(236)
        plt.title('Sobel thresholded mask')
        plt.imshow(sobel_mag_mask, cmap='gray')

        plt.tight_layout()
        if show:
            plt.show(block=False)

        plt.savefig('output_images/%s_edge_images.jpg' % image_name)
        plt.close()

    return lane_image


def lane_detection(image_name, image, KK, Kc, show=False, output_images=False):
    """ Main lane detection function
    Arguments:
        image_name {[type]} -- [description]
        image {[type]} -- [description]
        KK {[type]} -- [description]
        Kc {[type]} -- [description]

    Keyword Arguments:
        show {bool} -- [description] (default: {False})
    """

    homography = get_homography()

    image = cv2.undistort(image, KK, Kc)

    warped_image = preprocess_image(image, image_name, show, output_images)

    # Sliding window lane searching
    left_poly, right_poly = search_lane(image_name, warped_image, KK, show=show, output_images=output_images)

    homography_inv = cv2.getPerspectiveTransform(WARPED_ROI_CORNERS, ROI_CORNERS)
    y_range = np.arange(0, warped_image.shape[0])
    left_lane_corners = None
    if left_poly is not None:
        left_curvature = measure_curvature_pixels(left_poly, homography.shape[0])
        log.debug('left_curvature=%s', left_curvature)
        warped_left_lane_corners = np.vstack([np.polyval(left_poly, y_range), y_range]).T
        left_lane_corners = cv2.perspectiveTransform(warped_left_lane_corners.reshape(1, -1, 2), homography_inv).reshape(-1, 2)

    right_lane_corners = None
    if right_poly is not None:
        right_curvature = measure_curvature_pixels(right_poly, homography.shape[0])
        log.debug('right_curvature=%s', right_curvature)
        warped_right_lane_corners = np.vstack([np.polyval(right_poly, y_range), y_range]).T
        right_lane_corners = cv2.perspectiveTransform(warped_right_lane_corners.reshape(1, -1, 2), homography_inv).reshape(-1, 2)

    detection_overlap = np.zeros_like(image)
    if left_lane_corners is not None and right_lane_corners is not None:
        detection_overlap = cv2.fillPoly(detection_overlap, np.vstack([left_lane_corners, right_lane_corners[::-1]]).astype(np.int32).reshape(1, -1, 2), color=[0, 255, 0])
        detection_overlap = cv2.polylines(detection_overlap, left_lane_corners.astype(np.int32).reshape(1, -1, 2), False, color=[255, 0, 0], thickness=5)
        detection_overlap = cv2.polylines(detection_overlap, right_lane_corners.astype(np.int32).reshape(1, -1, 2), False, color=[0, 0, 255], thickness=5)
    overlay_image = cv2.addWeighted(image, 1.0, detection_overlap, 0.5, 0.0)

    if output_images:
        if show:
            plt.clf()
            plt.imshow(overlay_image)
            plt.show(block=False)

    return overlay_image


def lane_detections(images, KK, Kc, output_path, show=False, output_images=False):
    for index, (image_name, image) in enumerate(images.items()):
        result_image = lane_detection(image_name, image, KK, Kc, show=show, output_images=output_images)
        result_image_path = os.path.join(output_path, image_name + '_result.jpg')
        if output_images:
            cv2.imwrite(result_image_path, cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))


def lane_detection_in_video(KK, Kc, input_path, video_name, output_path, output_images=False):
    cap = cv2.VideoCapture(os.path.join(input_path, video_name))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 30.0, (frame_width, frame_height))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_image = lane_detection('%s_%s' % (video_name, frame_id), frame, KK, Kc, output_images=output_images)
        frame_id += 1
        writer.write(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()


if __name__ == '__main__':
    plt.ion()
    script_path = os.path.dirname(os.path.realpath(__file__))
    test_image_input_path = os.path.join(script_path, '..', 'test_images')
    test_images = load_images(test_image_input_path)

    camera_cal_image_path = os.path.join(script_path, '..', 'camera.json')
    video_path = os.path.join(script_path, '..')

    import ujson
    with open(camera_cal_image_path, 'r') as f:
        data = ujson.load(f)
        KK = np.array(data['KK'])
        Kc = np.array(data['Kc'])

    test_images_output_path = os.path.join(script_path, '../output_images')
    if not os.path.exists(test_images_output_path):
        os.mkdir(test_images_output_path)

    lane_detections(test_images, KK, Kc, output_path=test_images_output_path, show=False, output_images=True)

    video_ouput_path = test_images_output_path
    lane_detection_in_video(KK, Kc, video_path, 'project_video.mp4', video_ouput_path)
    lane_detection_in_video(KK, Kc, video_path, 'challenge_video.mp4', video_ouput_path, output_images=True)
    lane_detection_in_video(KK, Kc, video_path, 'harder_challenge_video', video_ouput_path)

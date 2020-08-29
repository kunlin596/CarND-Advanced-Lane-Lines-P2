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
ROI_CORNERS = np.float32([[IMAGE_SHAPE[1] / 2 - 62, IMAGE_SHAPE[0] / 2 + 100],
                          [IMAGE_SHAPE[1] / 6 - 17, IMAGE_SHAPE[0]],
                          [IMAGE_SHAPE[1] * 5 / 6 + 67, IMAGE_SHAPE[0]],
                          [IMAGE_SHAPE[1] / 2 + 62, IMAGE_SHAPE[0] / 2 + 100]])


WARPED_ROI_CORNERS = np.float32([[IMAGE_SHAPE[1] / 4, 0],
                                [IMAGE_SHAPE[1] / 4, IMAGE_SHAPE[0]],
                                [IMAGE_SHAPE[1] * 3 / 4, IMAGE_SHAPE[0]],
                                [IMAGE_SHAPE[1] * 3 / 4, 0]])

Y_METER_PER_PIXEL = 30 / 720
X_METER_PER_PIXEL = 3.7 / 700


def get_homography():
    src = ROI_CORNERS
    dst = WARPED_ROI_CORNERS
    return cv2.findHomography(src, dst)


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


def search_lane(image, KK, window_size=None, show=False):
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
        window_size = (100, 20)

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
    lane_width_in_pixel = 596.1325

    # The distance between the 2 peaks should be roughly `lane_width_in_pixel`,
    # Use the 2nd peak is way small than the 1st peak, use 1st peak to fix the 2nd peak
    left_x = edges[left_max_bin: left_max_bin + 1].mean()
    rightX = edges[right_max_bin: right_max_bin + 1].mean()

    if right_max / left_max < 0.3:
        rightX = left_x + lane_width_in_pixel
    elif left_max / right_max < 0.3:
        left_x = rightX - lane_width_in_pixel

    # left_x = WARPED_ROI_CORNERS[0, 0]
    # rightX = WARPED_ROI_CORNERS[2, 0]

    window_height = window_size[1]
    current_y = image.shape[0] - window_height / 2
    current_left_x = int(left_x)
    current_right_y = int(rightX)

    def get_window_patch(currentX, current_y, window_size):
        corners = cv2.boxPoints(((currentX, current_y), window_size, 0)).astype(np.int32)
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
        patch = image[min_xy[1]: max_xy[1], min_xy[0]: max_xy[0]]
        return patch, min_xy, corners

    left_centers = []
    right_centers = []

    if show:
        plt.cla()
        plt.imshow(image, cmap='gray')
        plt.plot(WARPED_ROI_CORNERS[:, 0], WARPED_ROI_CORNERS[:, 1], 'r.')
        plt.show(block=False)

    current_window_size = (200, 20)

    while current_y > 0:
        left_patch, top_left_corner_left, left_corners = get_window_patch(current_left_x, current_y, current_window_size)
        right_patch, top_left_corner_right, right_corners = get_window_patch(current_right_y, current_y, current_window_size)

        valid_x = np.nonzero(left_patch)[1]
        x = sorted(valid_x)[len(valid_x) // 2] if len(valid_x) > 5 else np.nan
        left_center = np.array([x, left_patch.shape[0] / 2]) + top_left_corner_left

        valid_x = np.nonzero(right_patch)[1]
        x = sorted(valid_x)[len(valid_x) // 2] if len(valid_x) > 5 else np.nan
        right_center = np.array([x, right_patch.shape[0] / 2]) + top_left_corner_right

        left_centers.append(left_center)
        right_centers.append(right_center)

        if ~np.isnan(left_center[0]):
            current_left_x = int(left_center[0] + 0.5)

        if ~np.isnan(right_center[0]):
            current_right_y = int(right_center[0] + 0.5)

        # Move window up
        current_y -= window_size[1]

        if show:
            plt.plot(left_corners[:, 0][[0, 1, 2, 3, 0]], left_corners[:, 1][[0, 1, 2, 3, 0]], 'r')
            plt.plot(right_corners[:, 0][[0, 1, 2, 3, 0]], right_corners[:, 1][[0, 1, 2, 3, 0]], 'b')

    left_centers = np.array(left_centers)
    right_centers = np.array(right_centers)

    left_centers = left_centers[~np.isnan(left_centers[:, 0])]
    right_centers = right_centers[~np.isnan(right_centers[:, 0])]

    left_poly = None
    estimated_left_x = None
    if len(left_centers):
        left_poly = np.polyfit(left_centers[:, 1], left_centers[:, 0], 2)
        estimated_left_x = np.polyval(left_poly, np.arange(0, image.shape[0]))

    right_poly = None
    estimated_right_x = None
    if len(right_centers):
        right_poly = np.polyfit(right_centers[:, 1], right_centers[:, 0], 2)
        estimated_right_x = np.polyval(right_poly, np.arange(0, image.shape[0]))

    if show:
        plt.plot(left_centers[:, 0], left_centers[:, 1], 'r.')
        if estimated_left_x is not None:
            plt.plot(estimated_left_x, np.arange(0, image.shape[0]))
        plt.plot(right_centers[:, 0], right_centers[:, 1], 'g.')
        if estimated_right_x is not None:
            plt.plot(estimated_right_x, np.arange(0, image.shape[0]))
        plt.pause(0.001)
        plt.show(block=False)
        embed()

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
    log.debug('current image_name=%s', image_name)

    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if not show:
        plt.ioff()
    plt.figure(figsize=(30, 20))
    plt.suptitle('Channel comparison of difference color transforms', fontsize=32)
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.title('hls channel[%d]' % i)
        plt.imshow(hls_image[:, :, i], cmap='gray')
        plt.subplot(2, 3, i + 1 + 3)
        plt.imshow(hsv_image[:, :, i], cmap='gray')
        plt.title('hsv channel[%d]' % i)
    plt.tight_layout()
    if show:
        plt.show(block=False)

    if output_images:
        plt.savefig('output_images/%s_color_transform_comparison.jpg' % image_name)

    # Use S channel in HSL image since the lane color is more out-standing than others
    lane_image = hls_image[:, :, 2]

    # Undistort image
    undistorted_image = cv2.undistort(lane_image, KK, Kc)
    kernel_size = 3

    # Create sobel image
    grad_x = np.abs(cv2.Sobel(undistorted_image, cv2.CV_16S, 1, 0, kernel_size))
    grad_y = np.abs(cv2.Sobel(undistorted_image, cv2.CV_16S, 0, 1, kernel_size))
    sobel_image = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    canny_image = cv2.Canny(lane_image.astype(np.uint8), 155, 255)

    # Warp image
    homography = get_homography()
    warp_transform = homography[0]
    warped_canny_image = cv2.warpPerspective(canny_image, warp_transform, (canny_image.shape[1], canny_image.shape[0]))
    warped_image = cv2.warpPerspective(image, warp_transform, (canny_image.shape[1], canny_image.shape[0]))

    if not show:
        plt.ioff()

    plt.figure(figsize=(30, 20))
    plt.suptitle('Warped edge images', fontsize=32)

    plt.subplot(231)
    plt.title('Original color image with ROI corners')
    image2 = cv2.polylines(image, (ROI_CORNERS + 0.5).astype(np.int32).reshape(1, -1, 2), True, (255, 0, 0), 2)
    plt.imshow(image2)

    plt.subplot(232)
    plt.title('Sobel image (visualization only, not used)')
    plt.imshow(sobel_image, cmap='gray')

    plt.subplot(233)
    plt.title('Canne image')
    plt.imshow(canny_image, cmap='gray')

    plt.subplot(234)
    plt.title('Warped canny image')
    plt.imshow(warped_canny_image, cmap='gray')

    plt.subplot(235)
    plt.title('Warp color image with warped ROI corners')
    warped_image2 = cv2.polylines(warped_image, (WARPED_ROI_CORNERS + 0.5).astype(np.int32).reshape(1, -1, 2), True, (255, 0, 0), 2)
    plt.imshow(warped_image2)

    plt.tight_layout()
    if show:
        plt.show(block=False)

    if output_images:
        plt.savefig('output_images/%s_edge_images.jpg' % image_name)

    # Sliding window lane searching
    left_poly, right_poly = search_lane(warped_canny_image, KK)

    warp_transform_inv = np.linalg.inv(warp_transform)
    y_range = np.arange(0, warped_image.shape[0])
    left_lane_corners = None
    if left_poly is not None:
        left_x = np.polyval(left_poly, y_range)
        left_curvature = measure_curvature_pixels(left_poly, warp_transform.shape[0])
        log.debug('left_curvature=%s', left_curvature)
        warped_left_lane_corners = np.vstack([left_x, y_range]).T
        left_lane_corners = cv2.perspectiveTransform(warped_left_lane_corners.reshape(-1, 1, 2), warp_transform_inv).reshape(-1, 2)

    right_lane_corners = None
    if right_poly is not None:
        rightX = np.polyval(right_poly, y_range)
        right_curvature = measure_curvature_pixels(right_poly, warp_transform.shape[0])
        log.debug('right_curvature=%s', right_curvature)
        warped_right_lane_corners = np.vstack([rightX, y_range]).T
        right_lane_corners = cv2.perspectiveTransform(warped_right_lane_corners.reshape(-1, 1, 2), warp_transform_inv).reshape(-1, 2)

    detection_overlap = np.zeros_like(image)
    if left_lane_corners is not None and right_lane_corners is not None:
        detection_overlap = cv2.fillPoly(detection_overlap, np.vstack([left_lane_corners, right_lane_corners[::-1]]).astype(np.int32).reshape(1, -1, 2), color=[0, 255, 0])
        detection_overlap = cv2.polylines(detection_overlap, left_lane_corners.astype(np.int32).reshape(1, -1, 2), False, color=[255, 0, 0], thickness=3)
        detection_overlap = cv2.polylines(detection_overlap, right_lane_corners.astype(np.int32).reshape(1, -1, 2), False, color=[0, 0, 255], thickness=3)

    overlay_image = cv2.addWeighted(image, 1.0, detection_overlap, 0.5, 0.0)

    if show:
        plt.clf()
        plt.imshow(overlay_image)
        # plt.plot(left_lane_corners[:, 0], left_lane_corners[:, 1], 'r')
        # plt.plot(right_lane_corners[:, 0], right_lane_corners[:, 1], 'b')
        plt.show(block=False)
        embed()

    return overlay_image


def lane_detections(images, KK, Kc, output_path, show=False, output_images=False):
    for index, (image_name, image) in enumerate(images.items()):
        result_image = lane_detection(image_name, image, KK, Kc, show=show, output_images=output_images)
        result_image_path = os.path.join(output_path, image_name + '.jpg')
        cv2.imwrite(result_image_path, cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))


def lane_detection_in_video(KK, Kc, input_path, video_name, output_path):
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

        result_image = lane_detection('%s_%s' % (video_name, frame_id), frame, KK, Kc)
        frame_id += 1
        writer.write(result_image)

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

    # video_ouput_path = test_images_output_path
    # for video_name in ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']:
    #     lane_detection_in_video(KK, Kc, video_path, video_name, video_ouput_path)

    embed()

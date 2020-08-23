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


def get_homography():
    src = ROI_CORNERS
    dst = WARPED_ROI_CORNERS
    return cv2.findHomography(src, dst)


def load_images(imagePath):
    if not os.path.exists(imagePath):
        raise Exception('%s is not valid.' % (imagePath))

    images = {}
    imagePaths = glob.glob(os.path.join(imagePath, '*.jpg'))

    for index, path in enumerate(imagePaths):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        log.debug('loaded %s', path)
        imageName = os.path.basename(path).split('.')[0]
        images[imageName] = image
    return images


def search_lane(image, KK, windowSize=None, show=False):
    if windowSize is None:
        windowSize = (100, 20)

    y, x = np.nonzero(image[image.shape[0] // 2:, :])
    bins = int(image.shape[1] / 10 + 0.5)
    h, edges = np.histogram(x, bins=bins)

    medianIndex = len(h) // 2
    leftMaxBin = np.argmax(h[:medianIndex])
    leftMax = h[:medianIndex].max()
    rightMaxBin = np.argmax(h[medianIndex:]) + medianIndex
    rightMax = h[medianIndex:].max()

    # This value should be computed using camera KK and extrinsics,
    # here, it is estimated using straight_line.jpg
    laneWidthInPixel = 596.1325

    # The distance between the 2 peaks should be roughly `laneWidthInPixel`,
    # Use the 2nd peak is way small than the 1st peak, use 1st peak to fix the 2nd peak
    leftX = edges[leftMaxBin: leftMaxBin + 1].mean()
    rightX = edges[rightMaxBin: rightMaxBin + 1].mean()

    if rightMax / leftMax < 0.3:
        rightX = leftX + laneWidthInPixel
    elif leftMax / rightMax < 0.3:
        leftX = rightX - laneWidthInPixel

    # leftX = WARPED_ROI_CORNERS[0, 0]
    # rightX = WARPED_ROI_CORNERS[2, 0]

    windowHeight = windowSize[1]
    currentY = image.shape[0] - windowHeight / 2
    currentLeftX = int(leftX)
    currentRightX = int(rightX)

    def get_window_patch(currentX, currentY, windowSize):
        corners = cv2.boxPoints(((currentX, currentY), windowSize, 0)).astype(np.int32)
        minXY = corners.min(axis=0)
        maxXY = corners.max(axis=0)
        patch = image[minXY[1]: maxXY[1], minXY[0]: maxXY[0]]
        return patch, minXY, corners

    leftCenters = []
    rightCenters = []

    if show:
        plt.cla()
        plt.imshow(image, cmap='gray')
        plt.plot(WARPED_ROI_CORNERS[:, 0], WARPED_ROI_CORNERS[:, 1], 'r.')
        plt.show(block=False)

    currentWindowSize = (200, 20)

    while currentY > 0:
        leftPatch, topLeftCornerLeft, leftCorners = get_window_patch(currentLeftX, currentY, currentWindowSize)
        rightPatch, topLeftCornerRight, rightCorners = get_window_patch(currentRightX, currentY, currentWindowSize)

        validX = np.nonzero(leftPatch)[1]
        x = sorted(validX)[len(validX) // 2] if len(validX) > 5 else np.nan
        leftCenter = np.array([x, leftPatch.shape[0] / 2]) + topLeftCornerLeft

        validX = np.nonzero(rightPatch)[1]
        x = sorted(validX)[len(validX) // 2] if len(validX) > 5 else np.nan
        rightCenter = np.array([x, rightPatch.shape[0] / 2]) + topLeftCornerRight

        leftCenters.append(leftCenter)
        rightCenters.append(rightCenter)

        if ~np.isnan(leftCenter[0]):
            currentLeftX = int(leftCenter[0] + 0.5)

        if ~np.isnan(rightCenter[0]):
            currentRightX = int(rightCenter[0] + 0.5)

        # Move window up
        currentY -= windowSize[1]

        if show:
            plt.plot(leftCorners[:, 0][[0, 1, 2, 3, 0]], leftCorners[:, 1][[0, 1, 2, 3, 0]], 'r')
            plt.plot(rightCorners[:, 0][[0, 1, 2, 3, 0]], rightCorners[:, 1][[0, 1, 2, 3, 0]], 'b')

    leftCenters = np.array(leftCenters)
    rightCenters = np.array(rightCenters)

    leftCenters = leftCenters[~np.isnan(leftCenters[:, 0])]
    rightCenters = rightCenters[~np.isnan(rightCenters[:, 0])]

    leftPoly = None
    estimatedLeftX = None
    if len(leftCenters):
        leftPoly = np.polyfit(leftCenters[:, 1], leftCenters[:, 0], 2)
        estimatedLeftX = np.polyval(leftPoly, np.arange(0, image.shape[0]))

    rightPoly = None
    estimatedRightX = None
    if len(rightCenters):
        rightPoly = np.polyfit(rightCenters[:, 1], rightCenters[:, 0], 2)
        estimatedRightX = np.polyval(rightPoly, np.arange(0, image.shape[0]))

    if show:
        plt.plot(leftCenters[:, 0], leftCenters[:, 1], 'r.')
        if estimatedLeftX is not None:
            plt.plot(estimatedLeftX, np.arange(0, image.shape[0]))
        plt.plot(rightCenters[:, 0], rightCenters[:, 1], 'g.')
        if estimatedRightX is not None:
            plt.plot(estimatedRightX, np.arange(0, image.shape[0]))
        plt.pause(0.001)
        plt.show(block=False)
        embed()

    return leftPoly, rightPoly


def lane_detection(imageName, image, KK, Kc, show=False):
    log.debug('current imageName=%s', imageName)

    hlsImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if show:
        for i in range(3):
            plt.subplot(2, 3, i + 1)
            plt.imshow(hlsImage[:, :, i], cmap='gray')
            plt.subplot(2, 3, i + 1 + 3)
            plt.imshow(hsvImage[:, :, i], cmap='gray')

    # Use S channel in HSL image since the lane color is more out-standing than others
    laneImage = hlsImage[:, :, 2]

    # Undistort image
    undistortedImage = cv2.undistort(laneImage, KK, Kc)
    kernelSize = 3

    # Create sobel image
    gradX = np.abs(cv2.Sobel(undistortedImage, cv2.CV_16S, 1, 0, kernelSize))
    gradY = np.abs(cv2.Sobel(undistortedImage, cv2.CV_16S, 0, 1, kernelSize))
    sobelImage = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)

    cannyImage = cv2.Canny(laneImage.astype(np.uint8), 155, 255)

    # Warp image
    homography = get_homography()
    warpTransform = homography[0]
    warpedCannyImage = cv2.warpPerspective(cannyImage, warpTransform, (cannyImage.shape[1], cannyImage.shape[0]))
    warpedImage = cv2.warpPerspective(image, warpTransform, (cannyImage.shape[1], cannyImage.shape[0]))

    if show:
        plt.figure()
        plt.ion()
        plt.subplot(321)
        plt.imshow(image)
        plt.plot(ROI_CORNERS[:, 0], ROI_CORNERS[:, 1], 'r.')
        plt.subplot(322)
        plt.imshow(warpedImage)
        plt.plot(WARPED_ROI_CORNERS[:, 0], WARPED_ROI_CORNERS[:, 1], 'r.')
        plt.subplot(323)
        plt.imshow(cannyImage, cmap='gray')
        plt.subplot(324)
        plt.imshow(warpedCannyImage, cmap='gray')
        plt.subplot(325)
        plt.imshow(sobelImage, cmap='gray')
        plt.subplot(326)
        plt.imshow(cannyImage, cmap='gray')
        plt.pause(0.001)
        plt.show(block=False)

    # Sliding window lane searching
    leftPoly, rightPoly = search_lane(warpedCannyImage, KK)

    yRange = np.arange(0, warpedImage.shape[0])
    if leftPoly is not None:
        leftX = np.polyval(leftPoly, yRange)
    if rightPoly is not None:
        rightX = np.polyval(rightPoly, yRange)

    warpedLeftLaneCorners = np.vstack([leftX, yRange]).T
    warpedRightLaneCorners = np.vstack([rightX, yRange]).T

    warpTransformInv = np.linalg.inv(warpTransform)
    leftLaneCorners = cv2.perspectiveTransform(warpedLeftLaneCorners.reshape(-1, 1, 2), warpTransformInv).reshape(-1, 2)
    rightLaneCorners = cv2.perspectiveTransform(warpedRightLaneCorners.reshape(-1, 1, 2), warpTransformInv).reshape(-1, 2)

    detectionOverlap = np.zeros_like(image)
    detectionOverlap = cv2.fillPoly(detectionOverlap, np.vstack([leftLaneCorners, rightLaneCorners[::-1]]).astype(np.int32).reshape(1, -1, 2), color=[0, 255, 0])
    detectionOverlap = cv2.polylines(detectionOverlap, leftLaneCorners.astype(np.int32).reshape(1, -1, 2), False, color=[255, 0, 0], thickness=3)
    detectionOverlap = cv2.polylines(detectionOverlap, rightLaneCorners.astype(np.int32).reshape(1, -1, 2), False, color=[0, 0, 255], thickness=3)

    overlayImage = cv2.addWeighted(image, 1.0, detectionOverlap, 0.5, 0.0)

    show = True
    if show:
        plt.clf()
        plt.imshow(overlayImage)
        # plt.plot(leftLaneCorners[:, 0], leftLaneCorners[:, 1], 'r')
        # plt.plot(rightLaneCorners[:, 0], rightLaneCorners[:, 1], 'b')
        plt.pause(0.001)
        plt.show(block=False)
        embed()


def lane_detections(images, KK, Kc, show=False):
    for index, (imageName, image) in enumerate(images.items()):
        lane_detection(imageName, image, KK, Kc)


if __name__ == '__main__':
    plt.ion()
    sciptPath = os.path.dirname(os.path.realpath(__file__))
    testImagePath = os.path.join(sciptPath, '..', 'test_images')
    images = load_images(testImagePath)

    cameraDataPath = os.path.join(sciptPath, '..', 'camera.json')
    import ujson
    with open(cameraDataPath, 'r') as f:
        data = ujson.load(f)
        KK = np.array(data['KK'])
        Kc = np.array(data['Kc'])

    lane_detections(images, KK, Kc)
    embed()

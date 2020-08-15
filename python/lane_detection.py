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


def get_homography(image):
    imgShape = image.shape

    src = np.float32([[(imgShape[1] / 2) - 55, imgShape[0] / 2 + 100],
                      [((imgShape[1] / 6) - 10), imgShape[0]],
                      [(imgShape[1] * 5 / 6) + 60, imgShape[0]],
                      [(imgShape[1] / 2 + 55), imgShape[0] / 2 + 100]])

    dst = np.float32([[(imgShape[1] / 4), 0],
                      [(imgShape[1] / 4), imgShape[0]],
                      [(imgShape[1] * 3 / 4), imgShape[0]],
                      [(imgShape[1] * 3 / 4), 0]])

    roiMaskX = dst[::2, 0].astype(np.int32)
    return cv2.findHomography(src, dst), src, dst, roiMaskX


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


def search_lane(image, leftX, rightX, windowSize=None, show=False):
    if windowSize is None:
        windowSize = (200, 20)

    windowHeight = windowSize[1]
    currentY = image.shape[0]
    currentLeftX = int(leftX)
    currentRightX = int(rightX)

    halfWindowWidth = windowSize[0] // 2

    # embed()

    def get_window_patch(currentX, currentY):
        minX = max(currentX - halfWindowWidth, 0)
        minY = max(currentY - windowHeight, 0)
        maxX = min(currentX + halfWindowWidth, image.shape[1])
        maxY = currentY

        patch = image[minY: currentY, minX: maxX]

        topLeftCorner = np.array([minX, minY])
        bottomRightCorner = np.array([maxX, maxY])
        return patch, topLeftCorner, bottomRightCorner

    leftCenters = []
    rightCenters = []

    while currentY > 0:
        leftPatch, topLeftCornerLeft, bottomRightCornerLeft = get_window_patch(currentLeftX, currentY)
        rightPatch, topLeftCornerRight, bottomRightCornerRight = get_window_patch(currentRightX, currentY)

        validX = np.nonzero(leftPatch)[1]
        x = sorted(validX)[len(validX) // 2] if len(validX) > 200 else np.nan
        leftCenter = np.array([x, leftPatch.shape[0] / 2]) + topLeftCornerLeft

        validX = np.nonzero(rightPatch)[1]
        x = sorted(validX)[len(validX) // 2] if len(validX) > 200 else np.nan
        rightCenter = np.array([x, rightPatch.shape[0] / 2]) + topLeftCornerRight

        leftCenters.append(leftCenter)
        rightCenters.append(rightCenter)

        # Move window up
        currentY -= windowSize[1]
        currentLeftX -= 3
        currentRightX -= 3

        # if True:
        #     plt.subplot(131)
        #     plt.imshow(image, cmap='gray')
        #     plt.subplot(132)
        #     plt.imshow(leftPatch)
        #     plt.subplot(133)
        #     plt.imshow(rightPatch)
        # embed()

    leftCenters = np.array(leftCenters)
    rightCenters = np.array(rightCenters)

    leftCenters = leftCenters[~np.isnan(leftCenters[:, 0])]
    rightCenters = rightCenters[~np.isnan(rightCenters[:, 0])]

    leftPoly = np.polyfit(leftCenters[:, 1], leftCenters[:, 0], 2)
    rightPoly = np.polyfit(rightCenters[:, 1], rightCenters[:, 0], 2)

    estimatedLeftX = np.polyval(leftPoly, leftCenters[:, 1])
    estimatedRightX = np.polyval(rightPoly, rightCenters[:, 1])

    if show:
        plt.imshow(image, cmap='gray')
        plt.plot(leftCenters[:, 0], leftCenters[:, 1], 'r.')
        plt.plot(estimatedLeftX, leftCenters[:, 1])
        plt.plot(rightCenters[:, 0], rightCenters[:, 1], 'g.')
        plt.plot(estimatedRightX, rightCenters[:, 1])

    embed()


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

    show = False
    # Use S channel in HSL image since the lane color is more out-standing than others
    laneImage = hlsImage[:, :, 2]

    # Undistort image
    undistortedImage = cv2.undistort(laneImage, KK, Kc)
    kernelSize = 3

    # Create sobel image
    gradX = np.abs(cv2.Sobel(undistortedImage, cv2.CV_16S, 1, 0, kernelSize))
    gradY = np.abs(cv2.Sobel(undistortedImage, cv2.CV_16S, 0, 1, kernelSize))
    sobelImage = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)

    lowerBound = np.clip(np.percentile(sobelImage.flatten(), 99), 100, 255)
    cannyImage = cv2.Canny(sobelImage.astype(np.uint8), lowerBound, 255)

    # Warp image
    homography, srcCorners, dstCorners, roiMaskX = get_homography(image)
    warpTransform = homography[0]
    warpedImage = cv2.warpPerspective(cannyImage, warpTransform, (cannyImage.shape[1], cannyImage.shape[0]))

    # Optional masking
    # mask = np.zeros_like(warpedImage, dtype=np.bool)
    # mask[:, roiMaskX[0]: roiMaskX[1]] = True
    # warpedImage[~mask] = 0

    # Sliding window lane searching
    search_lane(warpedImage, leftX=roiMaskX[0], rightX=roiMaskX[1])

    if show:
        plt.subplot(121)
        plt.imshow(cannyImage, cmap='gray')
        plt.subplot(122)
        plt.imshow(warpedImage, cmap='gray')

    embed()


def lane_detections(images, KK, Kc, show=False):
    for index, (imageName, image) in enumerate(images.items()):
        lane_detection(imageName, image, KK, Kc)
    embed()


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

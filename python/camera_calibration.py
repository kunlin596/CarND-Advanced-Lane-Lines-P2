#!/usr/bin/env python3

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

from scipy.spatial.transform import Rotation as R

import logging
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

import matplotlib
matplotlib.use('TkAgg')

N_ROWS = 6
N_COLS = 9


def _generate_object_points(nImages, nRows=N_ROWS, nCols=N_COLS):
    objectPointsPerImage = np.zeros(shape=(nRows * nCols, 3), dtype=np.float32)
    objectPointsPerImage[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    objPoints = np.tile(objectPointsPerImage, reps=(nImages, 1)).reshape(nImages, nRows * nCols, 3)
    return objPoints


def load_images(imagePath):
    if not os.path.exists(imagePath):
        raise Exception('%s is not valid.' % (imagePath))

    images = {}
    imagePaths = glob.glob(os.path.join(imagePath, '*.jpg'))
    imagePaths.sort(key=lambda x: int(x[(x.find('calibration') + len('calibration')): x.rfind('.')]))

    for index, path in enumerate(imagePaths):
        image = cv2.imread(path)
        log.debug('loaded %s', path)
        imageName = os.path.basename(path).split('.')[0]
        images[imageName] = image

    return images


def detect_chess_board_corners(images, output_name, show=False):
    imagePoints = {}
    nImages = len(images)

    if not show:
        plt.ioff()

    nCols = 5
    nRows = int(np.round(nImages * 1.0 / nCols))
    plt.figure(figsize=(30.0, 20.0))
    plt.suptitle('Camera calibration - undistorted with detected pattern corners', fontsize=32)

    log.debug('found %d images', nImages)
    for index, (imageName, image) in enumerate(images.items()):
        if image is not None:
            grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayimage, (N_COLS, N_ROWS), None)

            if not ret:
                log.error('Chessborard corners not found in %s', imageName)
                continue

            corners = corners.reshape(-1, 2)
            imagePoints[imageName] = corners
            log.debug('detected corners shape, %s', corners.shape)

            plt.subplot(nRows, nCols, index + 1)
            plt.imshow(image)
            plt.plot(corners[:, 0], corners[:, 1], marker='.', markersize=3, linewidth=0, color='red')
            if show:
                plt.show(block=False)

    if show:
        embed()

    plt.tight_layout()
    plt.savefig('output_images/%s.png' % output_name)

    return imagePoints


def calibrate_camera(imagePath, show=False):
    images = load_images(imagePath)
    imagePoints = detect_chess_board_corners(images, output_name='undistort_images_with_detected_corners', show=show)
    imagePointsValues = np.array(list(imagePoints.values()))
    objectPoints = _generate_object_points(nImages=len(imagePointsValues))
    assert(imagePointsValues.shape[:-2] == objectPoints.shape[:-2])
    imageSize = list(images.values())[0].shape[:2]

    retvalue, KK, Kc, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objectPoints,
        imagePoints=imagePointsValues,
        imageSize=imageSize,
        cameraMatrix=None,
        distCoeffs=None
    )

    return KK, Kc, rvecs, tvecs


def get_camera_transforms(rvecs, tvecs):
    assert(len(rvecs) == len(tvecs))
    transforms = []
    for index in range(len(rvecs)):
        transform = np.eye(4)
        r = R.from_rotvec(rvecs[index].flatten())
        transform[:3, :3] = r.as_dcm()
        transform[:3, 3] = tvecs[index].flatten()
        transforms.append(transform)
    return np.array(transforms, copy=False)


def undistort_images(images, KK, Kc, show=False):
    undistortedImages = {}
    for name, image in images.items():
        undistortedImage = cv2.undistort(image, cameraMatrix=KK, distCoeffs=Kc)
        undistortedImages[name] = undistortedImage
        if show:
            plt.imshow(undistortedImage)
            plt.pause(0.3)
    return undistortedImages


if __name__ == '__main__':
    sciptPath = os.path.dirname(os.path.realpath(__file__))
    calibImagePath = os.path.join(sciptPath, '..', 'camera_cal')
    KK, Kc, rvecs, tvecs = calibrate_camera(calibImagePath, show=False)
    transforms = get_camera_transforms(rvecs, tvecs)

    import ujson
    with open(os.path.join(sciptPath, '..', 'camera.json'), 'w') as f:
        data = {
            'KK': KK.tolist(),
            'Kc': Kc.tolist()
        }
        ujson.dump(data, f, indent=2)

    # For testing
    images = load_images(calibImagePath)
    undistortedImages = undistort_images(images, KK, Kc, show=False)

    plt.ioff()
    plt.figure()
    plt.suptitle('Camera calibration result example')
    plt.subplot(121)
    plt.title('Before calibration')
    plt.imshow(images['calibration1'])
    plt.subplot(122)
    plt.title('After calibration')
    plt.tight_layout()
    plt.imshow(undistortedImages['calibration1'])
    plt.savefig('./output_images/camera_calibration_example.jpg')

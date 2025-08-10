import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scp
from cv24_lab2_2_utils import read_video, show_detection, orientation_histogram
import pickle
from part2_utils import *
from interest_points_detectors import gabor_detector
import os


def hog_descriptor(video, interest_points, nbins):
    """
    Compute the HOG descriptors of a video.

    """
    # gradients
    Lx, Ly, _ = video_gradients(video.astype(float))
    descriptors = []
    for point in interest_points:
        side = int(round(4 * point[3]))
        leftmost = int(max(0, point[0] - side))
        rightmost = int(min(video.shape[1] - 1, point[0] + side + 1))
        upmost = int(max(0, point[1] - side))
        downmost = int(min(video.shape[0] - 1, point[1] + side + 1))

        descriptor = orientation_histogram(
            Lx[upmost:downmost, leftmost:rightmost, int(point[2])],
            Ly[upmost:downmost, leftmost:rightmost, int(point[2])],
            nbins,
            np.array([side, side]),
        )
        descriptors.append(descriptor)
    return np.array(descriptors)


def hof_descriptor(video, interest_points, nbins):
    """
    Compute the HOF descriptors of a video.

    """

    oflow = cv2.DualTVL1OpticalFlow_create(nscales=1)
    descriptors = []
    for point in interest_points:
        side = int(round(4 * point[3]))
        leftmost = int(max(0, point[0] - side))
        rightmost = int(min(video.shape[1] - 1, point[0] + side + 1))
        upmost = int(max(0, point[1] - side))
        downmost = int(min(video.shape[0] - 1, point[1] + side + 1))

        flow = oflow.calc(
            video[upmost:downmost, leftmost:rightmost, int(point[2] - 1)],
            video[upmost:downmost, leftmost:rightmost, int(point[2])],
            None,
        )
        descriptor = orientation_histogram(
            flow[..., 0], flow[..., 1], nbins, np.array([side, side])
        )
        descriptors.append(descriptor)
    return np.array(descriptors, dtype=np.float32)


def hog_hof_descriptor(video, interest_points, nbins):
    """
    Compute the HOG and HOF descriptors of a video.

    """
    hog = hog_descriptor(video, interest_points, nbins)
    hof = hof_descriptor(video, interest_points, nbins)
    return np.concatenate((hog, hof))


#
# def hog_hof_descriptor(detector_name):
#     filepath = "../../checkpoints/" + detector_name
#     with open(filepath + "_HOG_train", "rb") as file:
#         desc_hog_train = np.array(pickle.load(file))
#         print(desc_hog_train.shape)
#     with open(filepath + "_HOF_train", "rb") as file:
#         desc_hof_train = np.array(pickle.load(file))
#
#         print(desc_hof_train.shape)
#     desc_train = np.concatenate((desc_hog_train, desc_hof_train), axis=0)
#
#     output_filepath = "../../checkpoints/" + detector_name + "_HOG-HOF_train"
#     os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
#     with open(output_filepath, "wb") as file:
#         pickle.dump(desc_train, file)
#     #


if __name__ == "__main__":
    # hog_hof_descriptor("Harris")    # print(desc_hog_train.shape)
    # with open(filepath + "_HOG_test", "rb") as file:
    #     desc_hog_test = np.array(pickle.load(file))
    #
    #     print(desc_hog_test.shape)
    #    with open(filepath + "_HOF_test", "rb") as file:
    #     desc_hof_test = np.array(pickle.load(file))
    #
    # desc_test = np.concatenate((desc_hog_test, desc_hof_test), axis=0)
    # return desc_train, desc_test

    # filepath = "../../cv24_lab2_part2/running/person01_running_d1_uncomp.avi"
    # video = read_video(filepath, -1, 0)
    # gabor = lambda V: gabor_detector(video=V, sigma=4, tau=2, k=0.1, N=500)
    # interest_points = gabor(video)
    # t = hog_descriptor(video, interest_points, 10)
    # print(t.shape)
    filepath = "../../checkpoints/Harris_HOG_test"
    with open(filepath, "rb") as file:
        desc_hog_train = np.array(pickle.load(file))
    print(desc_hog_train.shape)

    filepath = "../../checkpoints/Harris_HOG_train"
    with open(filepath, "rb") as file:
        desc_hog_train = np.array(pickle.load(file), dtype=object)
    print(desc_hog_train.shape)
    desc = []
    for i in range(36):
        desc.append(desc_hog_train[i])
    desc = np.array(desc, dtype=object)
    print(desc.shape)
    #

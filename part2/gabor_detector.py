import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scp
from cv24_lab2_2_utils import read_video, show_detection
from io import BytesIO

from part2_utils import *

from interest_points_detectors import (
    gabor_detector,
    multiscale_gabor_detector,
    multiscale_harris_detector,
)
from histogram_descriptors import hog_descriptor

if __name__ == "__main__":
    filepath = "../../cv24_lab2_part2/boxing/person01_boxing_d2_uncomp.avi"

    filepath = "../../cv24_lab2_part2/walking/person04_walking_d1_uncomp.avi"
    video = read_video(filepath, -1, 0)
    video = video.copy()
    # #play_video(video)
    #
    # points = gabor_detector(video, 4, 1.5, 0.005, 500)
    # print(points)
    # show_detection(video, points)

    # filepath = "../../cv24_lab2_part2/walking/person04_walking_d1_uncomp.avi"
    # #:filepath = '../../cv24_lab2_part2/running/person01_running_d1_uncomp.avi'
    # video = read_video(filepath, -1, 0)
    # video = video.copy()
    # # play_video(video)
    #
    # points = gabor_detector(video, 4, 3.5, 0.2, 500)
    # print(points)
    # show_detection(video, points)
    #
    filepath = "../../cv24_lab2_part2/walking/person01_boxing_d2_uncomp.avi"
    #:filepath = '../../cv24_lab2_part2/running/person01_running_d1_uncomp.avi'
    video = read_video(filepath, -1, 0)
    video = video.copy()
    # play_video(video)
    #
    points = gabor_detector(video, 5, 3.5, 0.3, 500)
    print(points)
    show_detection(video, points, save_path="../../plots/part2")
    #
    # sigma = 2
    # tau = 1.5
    # s = 1.1
    # k = 0.4
    # N = 8
    #
    # sigmas = [sigma * s**i for i in range(N)]
    # #
    # # points = multiscale_gabor_detector(video, sigmas, tau, k, N=800)
    # # print(points)
    # #
    # # show_detection(video, points)
    #
    # points = multiscale_harris_detector(video, sigmas, tau, s=0.2, k=0.2, N=800)
    # print(points)
    #
    # show_detection(video, points)

    # filepath = "../../cv24_lab2_part2/walking/person04_walking_d1_uncomp.avi"
    # video = read_video(filepath, -1, 0)
    # video = video.copy()
    # #play_video(video)
    #
    # points = gabor_detector(video, 4, 1.5, 0.005, 500)
    # print(points)
    # show_detection(video, points)

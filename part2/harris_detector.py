import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as scp 
from cv24_lab2_2_utils import read_video, show_detection
from io import BytesIO

from part2_utils import *

from interest_points_detectors import harris_detector
if __name__ == "__main__":

    # filepath = '../../cv24_lab2_part2/boxing/person01_boxing_d2_uncomp.avi'
    # video = read_video(filepath, -1, 0)
    # video = video.copy()
    # #play_video(video)
    #
    # points = gabor_detector(video, 4, 1.5, 0.005, 500)
    # print(points)
    # show_detection(video, points)

    filepath = '../../cv24_lab2_part2/boxing/person01_boxing_d2_uncomp.avi'
    #filepath = '../../cv24_lab2_part2/running/person01_running_d1_uncomp.avi'
    filepath = '../../cv24_lab2_part2/walking/person04_walking_d1_uncomp.avi'
    video = read_video(filepath, -1, 0)
    video = video.copy()
    #play_video(video)

    points = harris_detector(video, 4, 1.5, 0.5, 0.05, 500)
    print(points)
    show_detection(video, points)

    # filepath = '../../cv24_lab2_part2/walking/person04_walking_d1_uncomp.avi'
    # video = read_video(filepath, -1, 0)
    # video = video.copy()
    # #play_video(video)
    #
    # points = gabor_detector(video, 4, 1.5, 0.005, 500)
    # print(points)
    # show_detection(video

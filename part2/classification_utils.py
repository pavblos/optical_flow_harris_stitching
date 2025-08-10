import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scp
import pickle
from tqdm import tqdm 

from cv24_lab2_2_utils import read_video, show_detection
from part2_utils import *

from interest_points_detectors import gabor_detector
from histogram_descriptors import hog_descriptor


def create_train_test(video_folders, train_file_path, test_file_path):
    def load_file_paths(file_path):
        with open(file_path, "r") as file:
            video_files = file.read().splitlines()
        return video_files

    def get_video_label(video_name):
        if "running" in video_name:
            return "running"
        elif "walking" in video_name:
            return "walking"
        elif "boxing" in video_name:
            return "boxing"
        else:
            return None

    def create_dataset(file_paths, folders):
        videos = []
        labels = []
        for video_name in file_paths:
            for folder in folders:
                filename = folder + video_name
                if os.path.isfile(filename):
                    videos.append(filename)
                    labels.append(get_video_label(video_name))
                    break
        return videos, labels

    # Load video file names from the training and test files
    train_files = load_file_paths(train_file_path)
    test_files = load_file_paths(test_file_path)

    # Create the training and test datasets with labels
    train_videos, train_labels = create_dataset(train_files, video_folders)
    test_videos, test_labels = create_dataset(test_files, video_folders)

    return train_videos, train_labels, test_videos, test_labels

def feature_extraction(train_data, test_data, detector, descriptor, savefile=None, loadfile=None, detector_name=None):
    def load_descriptors(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)

    # If loadfile is provided, load the descriptors from the file
    if loadfile is not None:
        desc_train = load_descriptors(loadfile + "_train")
        desc_test = load_descriptors(loadfile + "_test")
        return np.array(desc_train), np.array(desc_test)
    
    if detector_name is not None:
        filepath = "../../checkpoints/" + detector_name

        print("\tFeature Extraction of Training Data")
        desc_hog_train = load_descriptors(filepath + "_HOG_train")
        desc_hof_train = load_descriptors(filepath + "_HOF_train")
        desc_train = np.concatenate((desc_hog_train, desc_hof_train), axis=0)
       
        print("\tFeature Extraction of Testing Data")
        desc_hog_test = np.array(load_descriptors(filepath + "_HOG_test"))
        desc_hof_test = np.array(load_descriptors(filepath + "_HOF_test"))
        desc_test = np.concatenate((desc_hog_test, desc_hof_test), axis=0)
        
        if savefile is not None:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(savefile), exist_ok=True)
        
            with open(savefile + "_train", "wb") as file:
                pickle.dump(desc_train, file)
            with open(savefile + "_test", "wb") as file:
                pickle.dump(desc_test, file)

        return desc_train, desc_test

    print("\tFeature Extraction of Training Data")
    desc_train = []
    for video_file in tqdm(train_data, desc="Training Data"):
        video = read_video(video_file, -1, 0)
        # Detect interest points
        interest_points = detector(video)
        # Compute descriptors
        descriptors = descriptor(video, interest_points)
        desc_train.append(np.array(descriptors))

    print("\tFeature Extraction of Testing Data")
    desc_test = []
    for video_file in tqdm(test_data, desc="Testing Data"):
        video = read_video(video_file, -1, 0)
        # Detect interest points
        interest_points = detector(video)
        # Compute descriptors
        descriptors = descriptor(video, interest_points)
        desc_test.append(np.array(descriptors))

    if savefile is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        
        with open(savefile + "_train", "wb") as file:
            pickle.dump(desc_train, file)
        with open(savefile + "_test", "wb") as file:
            pickle.dump(desc_test, file)

    return np.array(desc_train), np.array(desc_test)
#
#
# def feature_extraction(train_data, test_data, detector, descriptor, savefile=None, loadfile=None, detector_name=None):
#     # If loadfile is provided, load the descriptors from the file
#     if loadfile is not None:
#         with open(loadfile + "_train", "rb") as file:
#             desc_train = pickle.load(file)
#         with open(loadfile + "_test", "rb") as file:
#             desc_test = pickle.load(file)
#         return np.array(desc_train), np.array(desc_test)
#     
#     if detector_name is not None:
#         filepath = "../../checkpoints/" + detector_name
#
#         print("\tFeature Extraction of Training Data")
#         with open(filepath + "_HOG_train", "rb") as file:
#             desc_hog_train = np.array(pickle.load(file))
#         with open(filepath + "_HOF_train", "rb") as file:
#             desc_hof_train = np.array(pickle.load(file))
#         desc_train = np.concatenate((desc_hog_train, desc_hof_train), axis=0)
#        
#         print("\tFeature Extraction of Testing Data")
#         with open(filepath + "_HOG_test", "rb") as file:
#             desc_hog_test = np.array(pickle.load(file))
#         with open(filepath + "_HOF_test", "rb") as file:
#             desc_hof_test = np.array(pickle.load(file))
#         desc_test = np.concatenate((desc_hog_test, desc_hof_test), axis=0)
#         
#         if savefile is not None:
#             # Ensure the directory exists
#             os.makedirs(os.path.dirname(savefile), exist_ok=True)
#         
#             with open(savefile + "_train", "wb") as file:
#                 pickle.dump(desc_train, file)
#             with open(savefile + "_test", "wb") as file:
#                 pickle.dump(desc_test, file)
#
#         return desc_train, desc_test
#
#     print("\tFeature Extraction of Training Data")
#     desc_train = []
#     for video_file in tqdm(train_data, desc="Training Data"):
#
#         video = read_video(video_file, -1, 0)
#         # Detect interest points
#         interest_points = detector(video)
#
#         # Compute descriptors
#         descriptors = descriptor(video, interest_points)
#         desc_train.append(np.array(descriptors))  # Extend the list with new descriptors
#
#     print("\tFeature Extraction of Testing Data")
#     desc_test = []
#     for video_file in tqdm(test_data, desc="Testing Data"):
#         
#         video = read_video(video_file, -1, 0)
#         # Detect interest points
#         interest_points = detector(video)
#
#         # Compute descriptors
#         descriptors = descriptor(video, interest_points)
#         desc_test.append(np.array(descriptors))
#
#     if savefile is not None:
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(savefile), exist_ok=True)
#         
#         with open(savefile + "_train", "wb") as file:
#             pickle.dump(desc_train, file)
#         with open(savefile + "_test", "wb") as file:
#             pickle.dump(desc_test, file)
#
#     return np.array(desc_train), np.array(desc_test)
#
#
if __name__ == "__main__":
    filepath = "../../cv24_lab2_part2/boxing/person01_boxing_d2_uncomp.avi"
    # filepath = '../../cv24_lab2_part2/running/person01_running_d1_uncomp.avi'
    # filepath = '../../cv24_lab2_part2/walking/person04_walking_d1_uncomp.avi'
    video = read_video(filepath, -1, 0)
    video = video.copy()
    # play_video(video)
    hog = lambda V, interest_points: hog_descriptor(V, interest_points, nbins=10)
    # hof = lambda V, interest_points, nbins: hof_descriptor(V, interest_points, nbins=10)
    # hog_hof = lambda V, interest_points, nbins: hog_hof_descriptors(V, interest_points, nbins=10)

    # detection functions
    # harris = lambda I: harris_stephens_detector(video=V, sigma=4, tau=2, s=0.2, k=0.1, N=500)
    gabor = lambda V: gabor_detector(video=V, sigma=4, tau=3.5, k=0.2, N=500)

    points = feature_extraction(
        video, gabor, hog, savefile="../../checkpoints/person01_boxing_d2_uncomp"
    )
    points = feature_extraction(
        video, gabor, hog, loadfile="../../checkpoints/person01_boxing_d2_uncomp"
    )
    print(points.shape)

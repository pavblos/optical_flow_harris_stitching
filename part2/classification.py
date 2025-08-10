import os 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import io 
from tqdm import tqdm 

from interest_points_detectors import harris_detector, harris_detector_smooth, gabor_detector, multiscale_harris_detector, multiscale_gabor_detector
from histogram_descriptors import hog_descriptor, hof_descriptor, hog_hof_descriptor
from classification_utils import *
from cv24_lab2_2_utils import bag_of_words, svm_train_test

if __name__ == '__main__':
   
    # using lambda as a wrapper for the used functions
    # decription functions
    hog = lambda V, interest_points: hog_descriptor(V, interest_points, nbins=10)
    hof = lambda V, interest_points: hof_descriptor(V, interest_points, nbins=10)
    hog_hof = lambda V, interest_points: hog_hof_descriptor(V, interest_points, nbins=10)

    # detection functions
    harris = lambda V: harris_detector(video=V, sigma=4, k=0.1, N=500)
    gabor = lambda V: gabor_detector(video=V, sigma=4, tau=2, k=0.1, N=500)
    multiscale_harris = lambda V: multiscale_harris_detector(video=V, sigma=3,tau=2, k=0.1,s=0.2, N=500, scale=1.2, num_scales=4)
    multiscale_gabor = lambda V: multiscale_gabor_detector(video=V, sigma=3, tau=2, k=0.1, N=500, scale=1.2, num_scales=4)


    descriptors = {
                    'HOG' : hog, 
                    'HOF' : hof, 
                   'HOG-HOF' : hog_hof
                   }
    detectors = {
                #'Harris' :                 harris,
                 #'Gabor' :                  gabor,
                'Multiscale-Harris' :      multiscale_harris, 
                #'Multiscale-Gabor' :       multiscale_gabor, 
                }
    
    video_folders = ["../../cv24_lab2_part2/running/", "../../cv24_lab2_part2/walking/", "../../cv24_lab2_part2/boxing/"]
    train_file_path = "training_set.txt"
    test_file_path = "test_set.txt"

    D = 50 

    print("Creating train and test set")
    train_data, train_labels, test_data, test_labels = create_train_test(video_folders, train_file_path, test_file_path)
    # Pre-compute the FeatureExtraction for every detector and descriptor 
    print("Feature Extraction")
    for detector_name, detector_func in detectors.items():
        for descriptor_name, descriptor_func in descriptors.items():
            filename = '../../checkpoints/' + detector_name + '_' + descriptor_name 
            if os.path.exists(filename+"_train") and os.path.exists(filename+"_test"):
                print("Files '{}', '{} already exist.".format(filename+"_train", filename+"_test"))
            else:
                print("Feature Extraction for {} detector with {} descriptor".format(detector_name, descriptor_name))
                desc_train, desc_test = feature_extraction(train_data, test_data, detector_func, descriptor_func, savefile=filename)
                # if descriptor_name == "HOG-HOF":
                #     desc_train, desc_test = feature_extraction(train_data, test_data, detector_func, descriptor_func, savefile=filename, detector_name=detector_name)
                # else:
                #     desc_train, desc_test = feature_extraction(train_data, test_data, detector_func, descriptor_func, savefile=filename)

    accs = []
    preds = []
    print("Classification")
    for detector_name, detector_func in detectors.items():
        for descriptor_name, descriptor_func in descriptors.items():
            print("Loading Descriptors")
            filename = '../../checkpoints/' + detector_name + '_' + descriptor_name 
            desc_train, desc_test = feature_extraction(train_data, test_data, detector_func, descriptor_func, loadfile=filename)
            print(desc_train.shape, desc_test.shape) 
            print("Creating Bag-of-Words for: ", detector_name, descriptor_name)
            bow_train, bow_test = bag_of_words(desc_train, desc_test, num_centers=3)
            
            print("Using SVM")
            accuracy, pred = svm_train_test(bow_train, train_labels, bow_test, test_labels)
            
            print('Accuracy for {} detector with {} descriptor: {:.3f}%'.format(detector_name, descriptor_name, 100.0*accuracy))
            accs.append(accuracy)
            preds.append(preds)

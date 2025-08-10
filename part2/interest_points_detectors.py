import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as scp 
from cv24_lab2_2_utils import read_video, show_detection

from part2_utils import *

def interest_points(response, num, threshold, scale):
    # Cornerness Condition 
    cond = (response > threshold * np.max(response.flatten())) 
    x, y, t = np.where(cond)
    points = np.column_stack((y,x,t, scale*np.ones(len(x))))
   
    response_indices = response[x,y,t]
    sorted_indices = np.argsort(response_indices)[::-1]

    top_points = points[sorted_indices[:num]]

    return top_points

def harris_detector(video, sigma, k, N):
    """
    Harris Detector
    """
    video = video.astype(float)/video.max()
    
    Lx, Ly, Lt = video_gradients(video)
    Lxx, Lxy, Lxt = video_gradients(Lx)
    _, Lyy, Lyt = video_gradients(Ly)
    _, _, Ltt = video_gradients(Lt)

    trace = Lxx + Lyy + Ltt
    det = Lxx * (Lyy * Ltt - Lyt * Lyt) - Lxy * (Lxy * Ltt - Lxt * Lyt) + Lxt * (Lxy * Lyt - Lyy * Lxt)

    response = abs(det - k * trace ** 3)
    
    points = interest_points(response, num=N, threshold=k, scale=sigma)
    return points

def harris_detector_smooth(video, sigma, tau, s, k, N):
    """
    Harris Detector with Gaussian filtering
    """
    space_kernel_size = int(np.ceil(3 * sigma) * 2 + 1)
    time_kernel_size = int(np.ceil(3 * tau) * 2 + 1)
    space_kernel = cv2.getGaussianKernel(space_kernel_size, sigma).T[0]
    time_kernel = cv2.getGaussianKernel(time_kernel_size, tau).T[0]

    video = video.astype(float)/video.max()
    video = video_smoothen(video, space_kernel, time_kernel)
    
    Lx, Ly, Lt = video_gradients_conv(video)

    space_kernel_size = int(np.ceil(3 * s * sigma) * 2 + 1)
    time_kernel_size = int(np.ceil(3 * s * tau) * 2 + 1)
    space_kernel = cv2.getGaussianKernel(space_kernel_size, s * sigma).T[0]
    time_kernel = cv2.getGaussianKernel(time_kernel_size, s * tau).T[0]
    
    Lxx = video_smoothen(Lx * Lx, space_kernel, time_kernel)
    Lyy = video_smoothen(Ly * Ly, space_kernel, time_kernel)
    Ltt = video_smoothen(Lt * Lt, space_kernel, time_kernel)
    Lxy = video_smoothen(Lx * Ly, space_kernel, time_kernel)
    Lxt = video_smoothen(Lx * Lt, space_kernel, time_kernel)
    Lyt = video_smoothen(Ly * Lt, space_kernel, time_kernel)
    
    trace = Lxx + Lyy + Ltt
    det = Lxx * (Lyy * Ltt - Lyt * Lyt) - Lxy * (Lxy * Ltt - Lxt * Lyt) + Lxt * (Lxy * Lyt - Lyy * Lxt)

    response = abs(det - k * trace ** 3)
    
    points = interest_points(response, num=N, threshold=k, scale=sigma)
    return points


def gabor_detector(video, sigma, tau, k, N):
    """
    Gabor Detector
    """ 
    video = video.astype(float)/video.max()

    # Smoothen video with G_σ
    #video = video_smoothen_space(video, sigma)

    # Define time variable in [-2τ, 2τ]
    t = np.linspace(-2*tau, 2*tau, int(4*tau+1))
    
    # Define omega
    omega = 4 / tau

    # Define the gabor filters and normalize them according to L1 norm
    h_even = np.cos(2 * np.pi * t * omega) * np.exp(-t**2 / (2 * tau**2))
    h_even /= np.linalg.norm(h_even, ord=1)

    h_odd = np.sin(2 * np.pi * t * omega) * np.exp(-t**2 / (2 * tau**2))
    h_odd /= np.linalg.norm(h_odd, ord=1)

    # Compute the response 
    response = (scp.convolve1d(video, h_even, axis=2))**2 + (scp.convolve1d(video, h_odd, axis=2))**2 
     
    points = interest_points(response, num=N, threshold=k, scale=sigma)
    return points

def multiscale_harris_detector(video, sigma, tau, s, k, N, scale, num_scales):
    """
    Multiscale Harris Detector

    """
    sigmas = [sigma * scale**i for i in range(num_scales)]
    points = []
    for sigma in sigmas:
        found = harris_detector(video, sigma, k, N)
        points.append(found)
    return log_metric_filter_points(video, points, tau, N)

def multiscale_gabor_detector(video, sigma, tau, k, N, scale, num_scales):
    """
    Multiscale Gabor Detector

    """
    sigmas = [sigma * scale**i for i in range(num_scales)]
    points = []
    for sigma in sigmas:
        found = gabor_detector(video, sigma, tau, k, N)
        points.append(found)
    return log_metric_filter_points(video, points, tau, N)




if __name__ == "__main__":
    sigma = 1
    space_size = int(2*np.ceil(3*sigma)+1)
    kernel = cv2.getGaussianKernel(space_size, sigma).T
    #print(kernel)
    #print()
    #print(kernel[0])

    filepath = '../../cv24_lab2_part2/running/person23_running_d2_uncomp.avi'
    video = read_video(filepath, -1, 0)
    video = video.copy()
    #play_video(video)

    # play_video_from_filepath(filepath)
    # #play_video(video)

    # points = gabor_detector(video, 4, 1.5, 0.005, 500)
    # print(points)
    # show_detection(video, points)

    # filepath = '../../cv24_lab2_part2/boxing/person01_boxing_d2_uncomp.avi'
    # video = read_video(filepath, -1, 0)
    # video = video.copy()
    #

    points = harris_detector(video, 4, 0.005, 500)
    #points = harris_detector(video, 4, 3.5, 2, 0.005, 500)
    print(points)
    show_detection(video, points)





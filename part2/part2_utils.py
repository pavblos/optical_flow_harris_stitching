import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scp
from cv24_lab2_2_utils import read_video, show_detection


def play_video_from_filepath(filepath):
    cap = cv2.VideoCapture(filepath)

    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame, ret)
        if ret:
            cv2.imshow("frame", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def play_video(video_file):
    # print(np.shape(video_file))
    for i in range(video_file.shape[2]):
        frame = video_file[:, :, i]
        cv2.imshow("frame", frame)
        cv2.waitKey(10)

def video_gradients(video):
    Lx, Ly, Lt = np.gradient(video)
    return Lx, Ly, Lt 

def video_gradients_conv(video):
    Lx = scp.convolve1d(video, np.array([-1, 0, 1]), axis=1)
    Ly = scp.convolve1d(video, np.array([-1, 0, 1]), axis=0)
    Lt = scp.convolve1d(video, np.array([-1, 0, 1]), axis=2)
    return Lx, Ly, Lt


def video_smoothen(video, space_kernel, time_kernel):
    video = scp.convolve1d(video, space_kernel, axis=0)
    video = scp.convolve1d(video, space_kernel, axis=1)
    video = scp.convolve1d(video, time_kernel, axis=2)
    return video


def video_smoothen_space(video, sigma):
    # define Gaussian kernel
    space_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
    # smoothen the video
    video = scp.convolve1d(video, kernel, axis=0)
    video = scp.convolve1d(video, kernel, axis=1)
    return video

def log_metric_filter_points(video, points_per_scale, tau, num_points):
    """
    Filters interest points according to the log metric
    
    """
    def log_metric(logs, itemsperscale, N):
        # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
        # returns the coordinates of the points that maximize
        # the log metric in a neighborhood of 3 scales
        # (prev scale), (curr scale), (next scale)
        final = []
        final_logs = []
        for index, items in enumerate(itemsperscale):
            logp = logs[max(index-1,0)]
            logc = logs[index]
            logn = logs[min(index+1,N-1)]
            for triplet in items:
                y, x, t = int(triplet[0]), int(triplet[1]), int(triplet[2])
                prev = logp[x, y, t]
                curr = logc[x, y, t]
                next = logn[x, y, t]
                if (curr >= prev) and (curr >= next):
                    final.append(triplet)
                    final_logs.append(curr)
        # get the points with top num_points log metric values
        if len(final) > num_points:
            indices = np.argsort(final_logs)[::-1]
            final_points = [final[i] for i in indices[:num_points]]
            return np.array(final_points)
        else:
            return np.array(final)
    v = video.copy()
    vnorm = v.astype(float)/video.max()
    # compute the laplacian of gaussian (log) metric
    logs = []
    time_size = int(2*np.ceil(3*tau)+1)
    time_kernel = cv2.getGaussianKernel(time_size, tau).T[0]
    # get the sigmas from the points
    sigmas = [item[0, 3] for item in points_per_scale]
    for sigma in sigmas:
        # define Gaussian kernel
        space_size = int(2*np.ceil(3*sigma)+1)
        space_kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
        v = video_smoothen(vnorm, space_kernel, time_kernel)
        # compute gradients
        Lx, Ly, _ = video_gradients(v)
        # compute second order derivatives
        _, Lyy, _ = video_gradients(Ly)
        Lxx, _, _ = video_gradients(Lx)
        # compute the log metric
        log = (sigma**2) * np.abs(Lxx + Lyy)
        logs.append(log)
    # find the points that maximize the log metric
    return log_metric(logs, points_per_scale, len(points_per_scale))




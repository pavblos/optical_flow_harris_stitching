import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import label
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
from mpl_toolkits.mplot3d import Axes3D

def save_image(image, name):
    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, f"{name}.png")
    
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_gaussian_3d(mean, covariance):
    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)

    x = np.linspace(80, 130, 100)
    y = np.linspace(130, 180, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    rv = multivariate_normal(mean, covariance)
    Z = rv.pdf(pos)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Cb')
    ax.set_ylabel('Cr')

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

    plt.title("Gaussian Skin Distribution")
    plt.close()

def compute_mean_covariance():
    mat_data = loadmat('cv24_lab2_part1\\skinSamplesRGB.mat')
    skin_samples_rgb = mat_data['skinSamplesRGB']
    skin_samples_bgr = cv2.cvtColor(skin_samples_rgb, cv2.COLOR_RGB2BGR)
    skin_samples_ycbcr = cv2.cvtColor(skin_samples_bgr, cv2.COLOR_BGR2YCrCb)
    Cb = skin_samples_ycbcr[:,:,2]
    Cr = skin_samples_ycbcr[:,:,1]

    mean_Cr = np.mean(Cr)
    mean_Cb = np.mean(Cb)
    mean = [mean_Cb, mean_Cr]

    covariance_matrix = np.cov(np.stack((Cr.ravel(), Cb.ravel()), axis=0))
    
    return mean, covariance_matrix

def fd(I, mu, cov):
    I_ycrcb = cv2.cvtColor(I, cv2.COLOR_RGB2YCR_CB)
    skin_image = np.full_like(I, 0)
    
    Cr = I_ycrcb[:,:,1]
    Cb = I_ycrcb[:,:,2]
    
    P_skin = multivariate_normal.pdf(np.dstack((Cb, Cr)), mean=mu, cov=cov)
    save_image(P_skin, "Skin probability image")
    
    skin_mask = (P_skin >= 0.0005).astype(np.uint8)
    save_image(skin_mask,"Original binary skin image")
    
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skin_opening = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, small_kernel)
    save_image(skin_opening,"Binary skin image after opening with elliptical kernel")
    
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    skin_closing = cv2.morphologyEx(skin_opening, cv2.MORPH_CLOSE, large_kernel)
    save_image(skin_closing, "Final skin image")
    
    labeled_array, num_features = label(skin_closing)
    print(num_features)
    
    bboxes = []
    
    for label_idx in range(1, num_features + 1):
        mask = (labeled_array == label_idx)
        bbox = cv2.boundingRect(mask.astype(np.uint8))
        bboxes.append(bbox)
    
    return bboxes

# Load image
rgb_frame = cv2.imread('cv24_lab2_part1\\1.png')
print(rgb_frame.shape)

rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
save_image(rgb_frame, "Original RGB frame")

mean, covariance = compute_mean_covariance()
print("Mean of Gaussian distribution: ",mean)
print("Covariance matrix of Gaussian distribution: ",covariance)
plot_gaussian_3d(mean, covariance)

skin_bboxes = fd(rgb_frame, mean, covariance)

areas = ["Face","(Her) right arm","(Her) left arm"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
for bbox, area in zip(skin_bboxes, areas):
    print(f"Bounding box for {area}:", bbox)

output_image = np.copy(rgb_frame)
for bbox, color in zip(skin_bboxes, colors):
    x, y, w, h = bbox
    cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)

save_image(output_image, "Frame with bounding boxes for skin areas")

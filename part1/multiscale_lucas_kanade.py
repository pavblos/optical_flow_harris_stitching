import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import label
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

from compute_total_shift import displ

from lucas_kanade import *

def double_scale(v):
    # Doubles input to upscale
    new_v = []
    for y,x in np.ndindex(v.shape):
        v[y,x]*=2
        new_v.append([[x,y]])
    return np.array(new_v)

def multiscale_lk(I1, I2, rho, epsilon, dx_0, dy_0, nr_of_scales):
        
    gaussian_kernel_1D = cv2.getGaussianKernel(int(2*np.ceil(rho)+1), 3)
    gaussian_kernel_2D = gaussian_kernel_1D @ (gaussian_kernel_1D).transpose()  # convolve with this gaussian to avoid aliasing
    
    I1 = cv2.filter2D(I1, -1, gaussian_kernel_2D)
    I2 = cv2.filter2D(I2, -1, gaussian_kernel_2D)
    I1_scales = [I1]
    I2_scales = [I2]
    
    for i in range(nr_of_scales):
        I1 = cv2.pyrDown(I1)
        I2 = cv2.pyrDown(I2)
        I1_scales.append(I1)
        I2_scales.append(I2)
    dx = np.zeros_like(I1_scales[-1])
    dy = np.zeros_like(I1_scales[-1])
        
    for I1, I2 in zip(I1_scales[::-1],I2_scales[::-1]): # Loop from smaller scales to bigger
        features = cv2.goodFeaturesToTrack(I2, 30, 0.02, 2, 5)
        if features is None:
          #  print("No features detected")
            continue
        features = np.int0(features)
        features = features.reshape((-1, 2))
        I1 = I1.astype(np.float64)/ np.max(I1)
        I2 = I2.astype(np.float64)/np.max(I2)
        
        # Resize d and double 
        dx = 2 * cv2.resize(dx.astype(np.float32), (I1.shape[1], I1.shape[0]))
        dy = 2 * cv2.resize(dy.astype(np.float32), (I1.shape[1], I1.shape[0]))

        dx, dy = lk(I1, I2, features, rho, epsilon, dx, dy)       
        
    return (dx), (dy)    
        
bbox = [75, 256, 78, 65]

image1_path = '.\\cv24_lab2_part1\\30.png'
#image2_path = '.\\cv24_lab2_part1\\2.png'

I1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#I2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

I2 = np.roll(I1, (5,5), axis = (0,1))
I1 = crop_image(I1, bbox)
I2 = crop_image(I2, bbox)

# Define parameters
rho = 5
epsilon = 0.01

# Call the lk function
dx, dy = multiscale_lk(I1, I2, rho, epsilon, np.zeros_like(I1), np.zeros_like(I1), 4)

plt.quiver(dx, dy, scale = 100, angles='xy',color='red')
plt.imshow(I1, cmap='gray')
plt.axis('off')
plt.savefig('output_images\motion_vectors_image_multi.jpg', bbox_inches='tight', pad_inches=0)
plt.show()

import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import label
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

from compute_total_shift import displ

def lk(I1, I2, features, rho, epsilon, dx_0, dy_0):
    # Generate 2D Gaussian kernel with
    n = int(2*np.ceil(3*rho)+1)
    mid = (n-1)//2
    
    gaussian_kernel_1D = cv2.getGaussianKernel(n, rho)
    gaussian_kernel_2D = gaussian_kernel_1D @ (gaussian_kernel_1D).transpose()  
    I1 = I1.astype(np.float64)/np.max(I1)
    I2 = I2.astype(np.float64)/np.max(I2)

    # Iterate over features
    for i, feature in enumerate(features):    
        x, y = feature.ravel()  # Extract (x, y) coordinates      
       
        # Focus on small area around the feature point to lower complexity
        i1 = I1[max(0, y-mid):min(y+mid, I1.shape[0]),
                           max(0, x-mid):min(x+mid, I1.shape[1])]
        i2 = I2[max(0, y-mid):min(y+mid, I2.shape[0]),
                        max(0, x-mid):min(x+mid, I2.shape[1])]

        dx_i = I1[max(0, y-mid):min(y+mid, dx_0.shape[0]),
                           max(0, x-mid):min(x+mid, dx_0.shape[1])]
        dy_i = I2[max(0, y-mid):min(y+mid, dy_0.shape[0]),
                        max(0, x-mid):min(x+mid, dy_0.shape[1])]

        I1_y, I1_x = np.gradient(I1)
                
        dx_i, dy_i = dx_0[y,x], dy_0[y,x]
        u = np.array([20,20])
        x_0, y_0 = np.meshgrid(np.arange(i1.shape[1]), np.arange(i1.shape[0]))
        
        for _ in range(30):   
            i1_shifted = map_coordinates(i1,[np.ravel(y_0+dy_i), np.ravel(x_0+dx_i)], order=1).reshape(i1.shape) # Shift I1 image by d_i=[dy_i,dx_i] with spline interpolation 
                                                                                                                # to deal with points between pixels. 
            E = i2 - i1_shifted  # Define error 

            A1 = map_coordinates(I1_x,[np.ravel(y_0+dy_i), np.ravel(x_0+dx_i)], order=1).reshape(i1.shape) # Shift I1_x by d_i=[dy_i,dx_i] 
            A2 = map_coordinates(I1_y,[np.ravel(y_0+dy_i), np.ravel(x_0+dx_i)], order=1).reshape(i2.shape) # Shift I1_y by d_i=[dy_i,dx_i] 

            # Compute elements of system
            A11 = A1*A1
            A22 = A2*A2
            A12 = A1*A2

            M11 = cv2.filter2D(src=A11, ddepth=-1, kernel=gaussian_kernel_2D)[i1.shape[0]//2,i1.shape[1]//2] + epsilon
            M12 = cv2.filter2D(src=A12, ddepth=-1, kernel=gaussian_kernel_2D)[i1.shape[0]//2,i1.shape[1]//2]
            M22 = cv2.filter2D(src=A22, ddepth=-1, kernel=gaussian_kernel_2D)[i1.shape[0]//2,i1.shape[1]//2] + epsilon

            A1E = A1*E
            v1 = cv2.filter2D(src=A1E, ddepth=-1, kernel=gaussian_kernel_2D)[i1.shape[0]//2,i1.shape[1]//2]
            A2E = A2* E
            v2 = cv2.filter2D(src=A2E, ddepth=-1, kernel=gaussian_kernel_2D)[i1.shape[0]//2,i1.shape[1]//2]  
        
            det = M11*M22 - M12*M12
            # Compute motion vector
            
            delta_x = (M22*v1 - M12*v2)/det
            delta_y = (M11*v2 - M12*v1)/det
            
            dx_i += delta_x
            dy_i += delta_y

            u = np.array([dx_i, dy_i])

        dx_0[y][x] = dx_i
        dy_0[y][x] = dy_i
        
    return dx_0, dy_0

def crop_image(image, box):
    x, y, width, height = box
    if len(image.shape) == 3:  # RGB image
        cropped_image = image[y:y+height, x:x+width, :]
    else:  # Grayscale image
        cropped_image = image[y:y+height, x:x+width]
    return cropped_image

def visualize_edges(I1, features):
    fig, ax = plt.subplots()

    # Show the image
    ax.imshow(I1)

    # Plot circles around each feature point
    for (x, y) in features:
        circle = plt.Circle((x, y), radius=2, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)

    # Show the plot with the image and circles
    plt.title("Points of interest to track")
    plt.show()
   
bbox = [151, 101, 76, 122]

image1_path = '.\\cv24_lab2_part1\\1.png'

# Read the images in grayscale and convert them to float32
I1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#I2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

I2 = I1[1:,5:]
I1 = crop_image(I1, bbox)
I2 = crop_image(I2, bbox)

rho = 5
epsilon = 0.01
dx_0 = np.zeros_like(I1)
dy_0 = np.zeros_like(I1)

features = cv2.goodFeaturesToTrack(I2, 30, 0.07, 1, 5)
print(features)
features = np.int0(features)

features = features.reshape((-1, 2))
print("features shape: ", features.shape)

# Call the lk function
dx, dy = lk(I1, I2, features, rho, epsilon, dx_0, dy_0)

# Compare results with OpenCV's built-in TVL1 Optical Flow method

    
optical_flow = cv2.DualTVL1OpticalFlow_create()

# Compute optical flow
flow = optical_flow.calc(I1.astype(np.uint8), I2.astype(np.uint8), None)

# Extract horizontal and vertical components of flow
flow_x = flow[:,:,0]
flow_y = flow[:,:,1]

# Create a meshgrid for plotting
h, w = flow.shape[:2]
x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

# Plot quiver plot
plt.figure()
#plt.imshow(I1, cmap='gray')
plt.quiver(x, y, -flow_x, -flow_y, scale=100, color='red')
plt.show()

I1 = cv2.imread(image1_path)
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)

I1 = crop_image(I1, bbox)

visualize_edges(I1, features)
print(displ(dx,dy, 0.5))
plt.quiver(-dx, -dy, scale = 100, angles='xy',color='red')
#plt.imshow(I1)
plt.axis('off')  # Optional: Turn off axis
#plt.title('Image with Motion Vectors')
plt.savefig('output_images\motion_vectors_image.jpg', bbox_inches='tight', pad_inches=0)
#plt.show()



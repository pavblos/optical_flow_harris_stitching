import numpy as np
import cv2
from matplotlib import pyplot as plt

from compute_total_shift import displ
from multiscale_lucas_kanade import *  # Assume these are correctly implemented
from lucas_kanade import *
bbox = [145, 110, 67, 148]  # (Her) right hand

rho = 5
epsilon = 0.01
dx = np.zeros((bbox[3], bbox[2]))  # Reset or update displacements
dy = np.zeros((bbox[3], bbox[2]))
image1_path = '.\\cv24_lab2_part1\\1.png'

img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
I2 = crop_image(img1, bbox)
plt.show()
for i in range(65):
    I1 = I2
    image2_path = f'.\\cv24_lab2_part1\\{i+2}.png'
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
   # I2 = I1[i:,i:]
    #I1 = crop_image(I1, bbox)
    I2 = crop_image(img2, bbox)

    features = cv2.goodFeaturesToTrack(I2, 60, 0.2, 1, 5)
    if features is not None:
        features = np.int0(features).reshape((-1, 2))
        #dx, dy = lk(I1, I2, features, rho, epsilon, np.zeros(I1.shape), np.zeros(I1.shape))
        
        dx, dy = multiscale_lk(I1, I2, rho, epsilon, np.zeros(I1.shape), np.zeros(I1.shape), 4)

        total_shift = displ(dx, dy, 0.4)
        bbox[0] += int(np.ceil(total_shift[0]))  # Update bounding box position
        bbox[1] += int(np.ceil(total_shift[1]))
    else:
        print("No features detected in image:", image1_path)

    #output_image = cv2.imread(image2_path)
    img2 = cv2.imread(image2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Save motion vectors
    
    plt.quiver(-dx, -dy, scale = 100, angles='xy',color='red')
    plt.axis('off')  # Optional: Turn off axis
    plt.savefig(f'box_face_video_multiscale\motion_vectors\motion_vectors_image{i+2}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()  
    
    cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    
    # Save frames with bboxes   
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.savefig(f"box_face_video_multiscale\\{i+2}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  
    
   
    print(bbox)
    
    
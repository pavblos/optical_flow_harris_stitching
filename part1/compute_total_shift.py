import numpy as np

def displ(d_x, d_y, threshold):
    # Computes total box shift by ignoring outliers
    
    # Compute the Euclidean distance squared for each (dx, dy) pair
    distances_squared = d_x**2 + d_y**2
    mean_distance = np.mean(distances_squared) 
    # Find the indices of non-outliers (distances >= 7)
    non_outliers_indices = np.where(distances_squared >= 0.5 * mean_distance)
    
    #print(non_outliers_indices[0])
    #print("dxxxxxxxx:",d_x)
    print("Initial number of (dx, dy): ", len(d_x))
    print("Number of (dx, dy) after energy thresholding: ", len(non_outliers_indices))
    
    if len(non_outliers_indices) == 0:  # Check if the list is empty
        print("No non-outlier detected")
        return 0, 0
    
    # Compute the mean of the non-outlier (dx, dy) values
    mean_dx = int(np.mean(d_x[non_outliers_indices]))
    mean_dy = int(np.mean(d_y[non_outliers_indices]))
    
    return mean_dx, mean_dy

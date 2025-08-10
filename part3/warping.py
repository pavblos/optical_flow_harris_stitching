import cv2
import numpy as np
import matplotlib.pyplot as plt


def forward_warping(H, img):
    # Get the height and width of the original image
    h, w = img.shape[:2]
    print("original height forward", h, w)
    # Generate a grid of points covering the original image
    y, x = np.mgrid[0:h, 0:w]

    # Convert the grid points to homogeneous coordinates
    points = np.array([x.ravel(), y.ravel(), np.ones_like(x.ravel())])

    # Apply the homography matrix H to the grid points
    new_points = np.dot(H, points)

    # Normalize the coordinates by dividing by the third coordinate
    new_points /= new_points[2]

    # Extract the new x and y coordinates
    new_x = new_points[0].reshape(h, w).astype(int)
    new_y = new_points[1].reshape(h, w).astype(int)

    # Find the minimum x and y coordinates
    x_min, y_min = np.min(new_x), np.min(new_y)

    # Shift the coordinates to make the minimum coordinate (0, 0)
    new_x -= x_min
    new_y -= y_min
    # print(x_min, y_min)
    # Compute the new height and width
    new_height, new_width = np.max(new_x) + 1, np.max(new_y) + 1

    # print("new height forward", new_height, new_width)
    # Create a new image with zeros
    new_img = np.zeros((new_height, new_width) + img.shape[2:], dtype=img.dtype)

    # Copy the original image onto the new image using the warped coordinates
    new_img[new_x, new_y] = img
    # Transpose the new image to switch x and y axes
    new_img = np.transpose(new_img, (1, 0, 2))

    # print(new_img.shape)
    # plt.imshow(new_img)
    # plt.show()

    return new_img, (x_min, y_min)


def inverse_warping(H, img):
    # Get the height and width of the original image
    h, w = img.shape[:2]

    # Generate a grid of points covering the output image
    new_height, new_width = img.shape[:2]
    y, x = np.mgrid[0:new_height, 0:new_width]

    # Convert the grid points to homogeneous coordinates
    points = np.array([x.ravel(), y.ravel(), np.ones_like(x.ravel())])

    # Apply the homography matrix H to the grid points
    new_points = np.dot(H, points)

    # Normalize the coordinates by dividing by the third coordinate
    new_points /= new_points[2]

    # Extract the new x and y coordinates
    new_x = new_points[0].reshape(new_height, new_width).astype(int)
    new_y = new_points[1].reshape(new_height, new_width).astype(int)

    # Find the minimum x and y coordinates
    x_min, y_min = np.min(new_x), np.min(new_y)

    # Shift the coordinates to make the minimum coordinate (0, 0)
    new_x_op = new_x - x_min
    new_y_op = new_y - y_min

    # Compute the new height and width
    new_h, new_w = np.max(new_x_op) + 1, np.max(new_y_op) + 1

    # Create a new image with zeros
    new_img = np.zeros((new_w, new_h) + img.shape[2:], dtype=img.dtype)

    # Find the inverse of the homography matrix H
    H_inv = np.linalg.inv(H)

    # Iterate over each pixel in the new image
    for i in range(new_w):
        for j in range(new_h):
            # Apply the inverse homography to find corresponding coordinates in original image
            orig_coords = np.dot(H_inv, [j + x_min, i + y_min, 1])
            orig_x, orig_y = (
                orig_coords[0] / orig_coords[2],
                orig_coords[1] / orig_coords[2],
            )

            # Check if the coordinates are within the bounds of the original image
            if 0 <= orig_x < w - 1 and 0 <= orig_y < h - 1:
                x_floor, y_floor = int(orig_x), int(orig_y)
                x_ceil, y_ceil = x_floor + 1, y_floor + 1

                dx, dy = orig_x - x_floor, orig_y - y_floor

                # Perform bilinear interpolation
                interpolated_value = (
                    (1 - dx) * (1 - dy) * img[y_floor, x_floor]
                    + dx * (1 - dy) * img[y_floor, x_ceil]
                    + (1 - dx) * dy * img[y_ceil, x_floor]
                    + dx * dy * img[y_ceil, x_ceil]
                )

                new_img[i, j] = interpolated_value
    # print(np.shape(new_img))
    # # new_img = np.transpose(new_img, (1, 0, 2))
    # plt.imshow(new_img)
    # plt.show()
    return new_img, (x_min, y_min)

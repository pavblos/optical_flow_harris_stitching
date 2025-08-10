import cv2
import numpy as np
import matplotlib.pyplot as plt 
from warping import forward_warping, inverse_warping


def extractFeatures(img, plot_flag=False, name=None):
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # creating a sift object and using detectandcompute() function to detect the keypoints and descriptor from the image
    sift_object = cv2.xfeatures2d.SIFT_create()
    keypoint, descriptor = sift_object.detectAndCompute(grayimage, None)

    # drawing the keypoints and orientation of the keypoints in the image and then displaying the image as the output on the screen
    keypoint_image = cv2.drawKeypoints(
        img,
        keypoint,
        None,
        color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    # if plot_flag:
    #     cv2.imshow("SIFT", keypoint_image)
    #     cv2.waitKey()
    # keypoint_image = cv2.drawKeypoints(img, keypoint, None, color=(0, 255, 0))
    if plot_flag:
        plt.imshow(keypoint_image)
        plt.axis('off')  # Hide axes
        plt.savefig("../../plots/part2/"+name, bbox_inches='tight', pad_inches=0)
        plt.show()
    return keypoint, descriptor



def projectionImage(H, img1, method="inverse_warping"):
    if method == "forward_warping":
        img1_warped, img1_topleft_coords = forward_warping(H, img1)
    if method == "inverse_warping":
        img1_warped, img1_topleft_coords = inverse_warping(H, img1)
    # img1_warped
    #print(img1_topleft_coords)
    return img1_warped, img1_topleft_coords

def mergeWarpedImages(img1_warped, img2, img1_topleft_coords):
    img1_height, img1_width = img1_warped.shape[:2]
    img2_height, img2_width = img2.shape[:2]

    # Calculate coordinates of the bounding box for the merged image
    min_x = min(img1_topleft_coords[0], 0)
    min_y = min(img1_topleft_coords[1], 0)
    max_x = max(img1_topleft_coords[0] + img1_width, img2_width)
    max_y = max(img1_topleft_coords[1] + img1_height, img2_height)

    merged_width = max_x - min_x
    merged_height = max_y - min_y

    merged_image = np.zeros((merged_height, merged_width, 3), dtype=img2.dtype)

    # Copy img1_warped onto merged_image
    img1_start_x = img1_topleft_coords[0] - min_x
    img1_start_y = img1_topleft_coords[1] - min_y
    img1_end_x = img1_start_x + img1_width
    img1_end_y = img1_start_y + img1_height
    merged_image[img1_start_y:img1_end_y, img1_start_x:img1_end_x] = img1_warped

    # Copy img2 onto merged_image
    img2_start_x = max(0, -min_x)
    img2_start_y = max(0, -min_y)
    img2_end_x = img2_start_x + img2_width
    img2_end_y = img2_start_y + img2_height
    merged_image[img2_start_y:img2_end_y, img2_start_x:img2_end_x] = np.maximum(merged_image[img2_start_y:img2_end_y, img2_start_x:img2_end_x], img2)

    # # merged_image = np.transpose(merged_image, (1, 0, 2))
    # plt.imshow(merged_image)
    # plt.show()

    return merged_image


def stitchImages(img1, img2, match_type="flann", plot_flag=False):
    # Step 1: Extract SIFT features and descriptors from both images
    print("Extracting Features")
    keypoints_img1, descriptors_img1 = extractFeatures(img1, plot_flag,"img1b_features.png")
    keypoints_img2, descriptors_img2 = extractFeatures(img2, plot_flag,"img2_features.png")


    # Step 2: Match features by applying FLANN-based matching
    print("Matching Features")
    if match_type == "bf":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_img1, descriptors_img2, 2)

    elif match_type == "flann":
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_img1, descriptors_img2, 2)

    # Lowe's ratio test
    RATIO_THRESHOLD = 0.7
    good_matches = []
    for m, n in matches:
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_matches.append(m)

    # Creating a boolean mask for the best matches
    matches_mask = np.zeros(len(good_matches)).astype(bool)
    for i, match in enumerate(good_matches):
        matches_mask[i] = True

    if plot_flag:
        img_matches = np.empty(
            (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
            dtype=np.uint8,
        )
        matched = cv2.drawMatches(
            img1,
            keypoints_img1,
            img2,
            keypoints_img2,
            good_matches,
            img_matches,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        plt.imshow(matched)
        plt.axis('off')  # Hide axes
        plt.savefig("../../plots/part2/matches.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        # cv2.imshow("", cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()

    # Extracting keypoints from good matches
    points1 = np.float32([keypoints_img1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    points2 = np.float32([keypoints_img2[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    print("Finding Homography")
    # Step 3: Find homography matrix using RANSAC
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    
    print("Warping the images")
    # Warp img1 onto img2 using the calculated homography matrix
    img1_warped, img1_topleft_coords = projectionImage(H, img1, "forward_warping")
    if plot_flag:
        plt.imshow(img1_warped)
        plt.axis('off')  # Hide axes
        plt.savefig("../../plots/part2/img1b_forward_warping.png", bbox_inches='tight', pad_inches=0)
        plt.show()
    img1_warped, img1_topleft_coords = projectionImage(H, img1)
    if plot_flag:
        plt.imshow(img1_warped)
        plt.axis('off')  # Hide axes
        plt.savefig("../../plots/part2/img1b_inverse_warping.png", bbox_inches='tight', pad_inches=0)
        plt.show()

    print("Merging the images")
    merged_image = mergeWarpedImages(img1_warped, img2, img1_topleft_coords)

    return merged_image

if __name__ == "__main__":
    img1 = cv2.imread("../../cv24_lab2_part3/img1b_ratio05.jpg", cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread("../../cv24_lab2_part3/img2_ratio05.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print("img1 shape = ", img1.shape)
    print("img2 shape = ", img2.shape)
    merged_image = stitchImages(img1, img2, plot_flag=True)
   

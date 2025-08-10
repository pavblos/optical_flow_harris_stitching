import cv2
import numpy as np
import matplotlib.pyplot as plt

from warping import *
from stitched_images_utils import stitchImages

if __name__ == "__main__":
    img1a = cv2.imread("../../cv24_lab2_part3/img1a_ratio05.jpg", cv2.IMREAD_COLOR)
    img1b = cv2.imread("../../cv24_lab2_part3/img1b_ratio05.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../../cv24_lab2_part3/img2_ratio05.jpg", cv2.IMREAD_COLOR)
    img3a = cv2.imread("../../cv24_lab2_part3/img3a_ratio05.jpg", cv2.IMREAD_COLOR)
    img3b = cv2.imread("../../cv24_lab2_part3/img3b_ratio05.jpg", cv2.IMREAD_COLOR)
    img4 = cv2.imread("../../cv24_lab2_part3/img4_ratio05.jpg", cv2.IMREAD_COLOR)
    img5 = cv2.imread("../../cv24_lab2_part3/img5_ratio05.jpg", cv2.IMREAD_COLOR)
    img6 = cv2.imread("../../cv24_lab2_part3/img6_ratio05.jpg", cv2.IMREAD_COLOR)

    img1a = cv2.cvtColor(img1a, cv2.COLOR_BGR2RGB)
    img1b = cv2.cvtColor(img1b, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3a = cv2.cvtColor(img3a, cv2.COLOR_BGR2RGB)
    img3b = cv2.cvtColor(img3b, cv2.COLOR_BGR2RGB)
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
    img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)

    img1b2 = stitchImages(img1b, img2)
    plt.imshow(img1b2)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img1b2.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img1a2 = stitchImages(img1a, img1b2)
    plt.imshow(img1a2)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img1a2.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img1a2 = stitchImages(img1a1b, img1b2)
    plt.imshow(img1a2)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img1a2.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img3a3b = stitchImages(img3b, img3a)
    plt.imshow(img3a3b)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img3a3b.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img23b = stitchImages(img3a3b, img2)
    plt.imshow(img23b)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img23b.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img13 = stitchImages(img23b, img1a2)
    plt.imshow(img13)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img13.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img45 = stitchImages(img4, img5)
    plt.imshow(img45)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img45.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img56 = stitchImages(img6, img5)
    plt.imshow(img56)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img56.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img46 = stitchImages(img45, img56)
    plt.imshow(img46)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img46.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    img16 = stitchImages(img46, img13)
    plt.imshow(img16)
    plt.axis("off")  # Hide axes
    plt.savefig("../../plots/part2/img16.png", bbox_inches="tight", pad_inches=0)
    plt.show()

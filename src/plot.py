from src.Utils import load_images
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2


def to_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(values * 255.))


if __name__ == "__main__":
    # face_image = cv2.imread('../dataset/testset/faces/cmu_0000.png').copy()
    # left, width = 8, 2
    # top, height = 3, 8
    #
    # face_image[top + height // 2:top + height, left:left + width] = 1
    # face_image[top:top + height // 2, left:left + width] = 255
    # plt.imshow(Image.fromarray(face_image))
    # cv2.imwrite('src/round1.png', face_image)
    # plt.show()
    # face_image = cv2.imread('../dataset/testset/faces/cmu_0000.png').copy()
    # left, width = 3, 2
    # top, height = 4, 4
    #
    # face_image[top:top + height, left:left + width // 2] = 1
    # face_image[top:top + height, left + width // 2 :left + width] = 255
    # plt.imshow(Image.fromarray(face_image))
    # cv2.imwrite('src/round1.png', face_image)
    # plt.show()
    # face_image = cv2.imread('../dataset/testset/faces/cmu_0000.png').copy()
    # left, width = 10, 2
    # top, height = 15, 4
    #
    # face_image[top:top + height // 3, left:left + width] = 1
    # face_image[top + height // 3:top + height // 3 * 2 , left:left + width] = 255
    # face_image[top + height // 3 * 2:top + height, left:left + width] = 1
    # plt.imshow(Image.fromarray(face_image))
    # cv2.imwrite('src/round5.png', face_image)
    # plt.show()
    face_image = cv2.imread('../dataset/testset/faces/cmu_0000.png').copy()
    left, width = 0, 2
    top, height = 0, 6

    face_image[top:top + height // 2, left:left + width // 2] = 1
    face_image[top + height // 2:top + height, left:left + width // 2] = 255
    face_image[top:top + height //2, left + width // 2: left + width] = 255
    face_image[top + height // 2:top + height, left + width // 2: left + width] = 1
    plt.imshow(Image.fromarray(face_image))
    cv2.imwrite('src/round10.png', face_image)
    plt.show()



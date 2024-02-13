# builtin imports

# external imports
import cv2
import numpy as np

# type hint imports
from typing import Tuple

# self-code imports


def animate_blur(image_path: str, mask: np.ndarray):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_copy = image.copy()
    width, height = image.shape[:2]
    mask_size = mask.shape[0]
    for x in range(width - mask_size):
        for y in range(height - mask_size):
            if x + mask_size <= width and y + mask_size <= height:
                image_copy[x : x + mask_size, y : y + mask_size] *= mask
                cv2.imshow("test", image_copy)
                cv2.waitKey(10)
    cv2.waitKey(0)


def generate_arrays(size) -> Tuple[np.ndarray, np.ndarray]:
    normal_array = np.ones((size, size), dtype=np.uint8)
    normal_array[1:, 1:] = 0
    reverse_array = np.ones((size, size), dtype=np.uint8)
    reverse_array[0:-1, 0:-1] = 0
    return normal_array, reverse_array


class RectDetect:
    def __init__(self, patch_size=4) -> None:
        self.patch_size = patch_size
        self.mask, self.inverse = generate_arrays(patch_size)

    def __repr__(self) -> str:
        return f"[RectDetect] with size {self.patch_size}"

    def mask_display(self) -> None:
        print(self.mask)
        print(self.inverse)


def main():
    recdec = RectDetect(10)
    input_image_path = "papers/test.jpg"
    animate_blur(input_image_path, recdec.mask)


if __name__ == "__main__":
    main()

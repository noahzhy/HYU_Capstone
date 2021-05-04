import numpy as np
from PIL import Image


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


if __name__ == "__main__":
    img_path = 'tests/duck.jpg'
    img = preprocess_image(Image.open(img_path))
    print(img)

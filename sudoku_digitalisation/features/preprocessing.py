import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset


dataset = load_dataset("Lexski/sudoku-image-recognition", split="train")


def load_data(path):
    """Loads a dataset from huggingface.com."""
    data = load_dataset(path)
    return data


def show_image(image):
    """Shows an image from the dataset."""
    plt.imshow(image, cmap="gray")
    plt.show()


def convert_to_grayscale(ds):
    """Converts the image to grayscale."""
    grayscale_image = ImageOps.grayscale(ds['image'])
    ds['image'] = grayscale_image
    return ds


def crop_to_bounding_box(ds, output_size=(450, 450)):
    """
    Crops the image to the bounding box. It does this by warping the image so it becomes a square.
    This also normalizes image sizes and makes images RGB.

    This is done using the "keypoints". They are a list of 8 values, which represent four coordinates.
    The keypoints are ordered: top-left, bottom-left, bottom-right, top-right.
    Some keypoints may be outside the bounds of the image.
    """
    keypoints = ds['keypoints']
    bounding_box = np.array([
        [keypoints[0], keypoints[1]],  # top left
        [keypoints[2], keypoints[3]],  # bottom left
        [keypoints[4], keypoints[5]],  # bottom right
        [keypoints[6], keypoints[7]]   # top right
    ], dtype=np.float32)

    desired_corner_points = np.array([
        [0, 0],  # top left
        [0, output_size[1] - 1],  # bottom left
        [output_size[0] - 1, output_size[1] - 1],  # bottom right
        [output_size[0] - 1, 0]  # top right
    ], dtype=np.float32)

    image = np.array(ds['image'].convert('RGB'))
    pers_transform_matrix = cv2.getPerspectiveTransform(bounding_box, desired_corner_points)
    warped_image = cv2.warpPerspective(image, pers_transform_matrix, output_size)
    ds['image'] = Image.fromarray(warped_image)

    return ds


def preprocess_dataset(ds_dict):
    """Preprocesses the dataset."""
    for split in ds_dict:
        print(split)
        ds_dict[split] = ds_dict[split].map(crop_to_bounding_box)
        ds_dict[split] = ds_dict[split].map(convert_to_grayscale)
        show_image(dataset_dict[split][0]["image"])
        print(dataset_dict[split][0]["keypoints"])


if __name__ == '__main__':
    dataset_dict = load_data("Lexski/sudoku-image-recognition")
    example_image = dataset_dict["train"][0]["image"]
    show_image(example_image)
    print(type(example_image))
    preprocess_dataset(dataset_dict)




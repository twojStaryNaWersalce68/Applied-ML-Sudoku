import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

def adaptive_histogram_equalization(ds):
    """
    Applies adaptive histogram equalization to the image.
    This is done using CLAHE algorithm.
    """
    image = np.array(ds['image'])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)
    ds['image'] = Image.fromarray(equalized_image)
    return ds

def hough_transform(ds):
    """
    Applies Hough transform to the image.
    This is done using cv2.HoughLinesP.
    """
    image = np.array(ds['image'])
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=500, maxLineGap=100)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ds['image'] = Image.fromarray(image)
    return ds

def split_image(ds):
    """
    Splits the image into 81 cells.
    """
    image = np.array(ds['image'])
    cells = []
    cell_size = (image.shape[0] // 9, image.shape[1] // 9)
    for i in range(9):
        for j in range(9):
            x_start = i * cell_size[0]
            x_end = (i + 1) * cell_size[0]
            y_start = j * cell_size[1]
            y_end = (j + 1) * cell_size[1]
            cell = image[x_start:x_end, y_start:y_end]
            cells.append(cell)
    ds['cells'] = cells
    return ds
    

def preprocess_dataset(ds_dict):
    """Preprocesses the dataset."""
    for split in ds_dict:
        print(split)
        ds_dict[split] = ds_dict[split].map(crop_to_bounding_box)
        ds_dict[split] = ds_dict[split].map(convert_to_grayscale)
        ds_dict[split] = ds_dict[split].map(adaptive_histogram_equalization)
        ds_dict[split] = ds_dict[split].map(split_image)

    return ds_dict


if __name__ == '__main__':
    dataset_dict = load_data("Lexski/sudoku-image-recognition")
    ds_dict = preprocess_dataset(dataset_dict)

    # Show one of the cells
    sample_cell = ds_dict['train'][0]['cells'][0]
    show_image(sample_cell)

    # To be implemented: cell images matched with labels




    
    
    




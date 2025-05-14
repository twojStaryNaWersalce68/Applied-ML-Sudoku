import matplotlib.pyplot as plt
import numpy as np
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
    """Converts the image to a rgb type and then to grayscale."""
    rgb_image = ds['image'].convert('RGB')
    grayscale_image = ImageOps.grayscale(rgb_image)
    ds['image'] = grayscale_image
    return ds


def preprocess_dataset(ds_dict):
    """Preprocesses the dataset."""
    for split in ds_dict:
        print(split)
        ds_dict[split] = ds_dict[split].map(convert_to_grayscale)
        show_image(dataset_dict[split][0]["image"])
        print(dataset_dict[split][0]["keypoints"])


if __name__ == '__main__':
    dataset_dict = load_data("Lexski/sudoku-image-recognition")
    example_image = dataset_dict["train"][0]["image"]
    show_image(example_image)
    print(type(example_image))
    preprocess_dataset(dataset_dict)


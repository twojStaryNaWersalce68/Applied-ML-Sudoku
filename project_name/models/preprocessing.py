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
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    ds_dict = load_data("Lexski/sudoku-image-recognition")
    show_image(ds_dict["train"][0]["image"])


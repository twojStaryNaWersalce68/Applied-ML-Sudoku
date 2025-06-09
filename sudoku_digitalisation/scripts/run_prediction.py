from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Union, Any, List
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from datasets import Dataset


def make_prediction(
        sample_sudoku: Union[Image.Image, Dict[str, Any]],
        preprocessor: DatasetPreprocessor,
        cnn: CNN
        ) -> List[int]:
    digit_dataset = preprocess(sample_sudoku, preprocessor)
    predictions = cnn.predict(digit_dataset)
    large_digits = find_labels_main(predictions)
    show_image(sample_sudoku, "Test Image")
    for idx, label in enumerate(large_digits):
        show_image(digit_dataset[idx], f"Label: {label}")
    return large_digits


def find_labels_main(predictions: np.ndarray) -> List[int]:
    labels = []
    for prediction in predictions:
        label = np.argmax(prediction)
        labels.append(int(label))
    return labels


def find_labels_candidate():
    '''    for idx in range(81):
    #idx = 3
        tiny_digits = CellSplitter.get_candidate_digits(digit_dataset[idx])
            #get_tiny_digits(digit_dataset[idx])) # 6 3 8/16 1 1

        if len(tiny_digits) > 0:
            predictions = cnn.predict(tiny_digits)
            cell_labels_binary = [0] * 10  # format for computer
            for i, prediction in enumerate(predictions):
                label = np.argmax(prediction)
                #print(label)
                #plt.imshow(tiny_digits[i], cmap="gray")
                #plt.title(label)
                #plt.show()
                if 0 <= label <= 9:
                    cell_labels_binary[label] = 1
            #print(cell_labels_binary)

            cell_labels = []  # format for humans
            for i in range(len(cell_labels_binary)):
                if cell_labels_binary[i] == 1:
                    cell_labels.append(i)
            print(idx, cell_labels)
            plt.imshow(digit_dataset[idx], cmap="gray")
            plt.title(f"{idx}: {cell_labels}")
            plt.show()

        else:
            print("no digits found")'''


def preprocess(sample_sudoku: Union[Image.Image, Dict[str, Any]], preprocessor: DatasetPreprocessor) -> List[Image.Image]:
    _, digit_dataset = preprocessor.sudoku_preprocessing(sample_sudoku)
    if isinstance(sample_sudoku, dict):
        digit_dataset = Dataset.from_list(digit_dataset)
        digit_dataset = digit_dataset['image']
    return digit_dataset


def show_image(image: Union[Image.Image, Dict[str, Any]], title: str = None) -> None:
    if isinstance(image, dict):
        image = image['image']
    plt.imshow(image, cmap="gray")
    if title:
        plt.title(title)
    plt.show()
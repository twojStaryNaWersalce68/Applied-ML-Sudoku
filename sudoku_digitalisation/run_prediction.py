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
) -> None:
    show_image(sample_sudoku, "Test Image")
    digit_dataset = preprocess(sample_sudoku, preprocessor)
    predictions = cnn.predict(digit_dataset)
    large_digits = find_labels_main(predictions)
    for idx, label in enumerate(large_digits):
        show_image(digit_dataset[idx], f"Label: {label}")


def find_labels_main(predictions: np.ndarray) -> List[int]:
    labels = []
    for prediction in predictions:
        label = np.argmax(prediction)
        labels.append(int(label))
    return labels


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
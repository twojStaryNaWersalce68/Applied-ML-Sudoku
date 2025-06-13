from PIL import Image
import numpy as np
from typing import Dict, Union, Any, List
from sudoku_digitalisation.features.cell_splitter import CellSplitter
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor, SudokuPreprocessor
from datasets import Dataset


##### Make sure not to do show image in here, otherwise you'll break fastAPI! #####
def make_prediction(
        sample_sudoku: Union[Image.Image, Dict[str, Any]],
        preprocessor: DatasetPreprocessor,
        cnn: CNN,
        binary: bool = False
        ) -> Union[List[int], List[List[int]]]:
    '''
    Full prediction pipeline for a sudoku
    '''
    digit_dataset = preprocess(sample_sudoku, preprocessor)
    predictions = cnn.predict(digit_dataset)
    large_digits = find_labels_main(predictions)
    all_digits = find_labels_candidate(large_digits, digit_dataset, cnn, binary)
    return large_digits, all_digits


def find_labels_main(predictions: np.ndarray) -> List[int]:
    '''
    Finds the labels of solution digits in the cells
    '''
    labels = []
    for prediction in predictions:
        label = np.argmax(prediction)
        labels.append(int(label))
    return labels


def find_labels_candidate(
        cell_labels: List[int],
        cell_images: List[Image.Image],
        cnn: CNN,
        binary: bool = False
        ) -> List[List[int]]:
    '''
    Finds the labels of the candidate digits in the empty cells and adds them to the dataset.
    '''
    binary_cell_labels = []
    human_cell_labels = []
    for idx, cell_img in enumerate(cell_images):
        binary_cell = [0] * 10
        label = cell_labels[idx]
        if label == 0:
            cand_cell_labels = []
            candidate_digits = CellSplitter.get_candidate_digits(cell_img)
            if len(candidate_digits) > 0:
                predictions = cnn.predict(candidate_digits)
                for prediction in predictions:
                    candidate_label = np.argmax(prediction)
                    if 0 < candidate_label <= 9:
                        binary_cell[candidate_label] = 1
                for jdx in range(len(binary_cell)):
                    if binary_cell[jdx] == 1:
                        cand_cell_labels.append(jdx)
            if len(cand_cell_labels) > 0:
                human_cell_labels.append(cand_cell_labels)
            else:
                human_cell_labels.append([0])
        else:
            human_cell_labels.append([label])
            binary_cell[0] = 1  # marks the cell as solved (has a large digit)
            binary_cell[label] = 1  # marks the digit
        binary_cell_labels.append(binary_cell)
    if binary:
        return binary_cell_labels
    return human_cell_labels


def preprocess(
        sample_sudoku: Union[Image.Image, Dict[str, Any]],
        preprocessor: Union[DatasetPreprocessor, SudokuPreprocessor]
        ) -> List[Image.Image]:
    '''
    Preprocesses the given sudoku image.
    '''
    _, digit_dataset = preprocessor.sudoku_preprocessing(sample_sudoku)
    if isinstance(sample_sudoku, dict):
        digit_dataset = Dataset.from_list(digit_dataset)
        digit_dataset = digit_dataset['image']
    return digit_dataset


from PIL import Image
import numpy as np
from typing import Dict, Tuple, Union, Any, List
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor, SudokuPreprocessor
from datasets import Dataset


##### Make sure not to do show image in here, otherwise you'll break fastAPI! #####
def make_prediction(
        sample_sudoku: Union[Image.Image, Dict[str, Any]],
        preprocessor: Union[DatasetPreprocessor, SudokuPreprocessor],
        cnn: CNN
        ) -> Union[List[int], List[List[int]]]:
    '''
    Full prediction pipeline for a sudoku
    '''
    digit_dataset = preprocess(sample_sudoku, preprocessor)
    predictions = cnn.predict(digit_dataset)
    large_digits = find_labels_main(predictions)
    all_digits = cnn.predict_candidate(large_digits, digit_dataset)
    return all_digits


def find_labels_main(predictions: np.ndarray) -> List[int]:
    '''
    Finds the labels of solution digits in the cells
    '''
    labels = []
    for prediction in predictions:
        label = np.argmax(prediction)
        labels.append(int(label))
    return labels

 
def preprocess(sample_sudoku: Union[Image.Image, Dict[str, Any]], preprocessor: Union[DatasetPreprocessor, SudokuPreprocessor]) -> List[Image.Image]:
    '''
    Preprocesses the given sudoku image.
    '''
    _, digit_dataset = preprocessor.sudoku_preprocessing(sample_sudoku)
    if isinstance(sample_sudoku, dict):
        digit_dataset = Dataset.from_list(digit_dataset)
        digit_dataset = digit_dataset['image']
    return digit_dataset


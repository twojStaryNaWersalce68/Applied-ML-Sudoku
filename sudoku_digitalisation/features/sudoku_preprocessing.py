import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from PIL import Image
from typing import Dict, Tuple, Union, Any
from sudoku_digitalisation.features.image_operations import ImageConverter, ImageCropper
from sudoku_digitalisation.features.edge_detector import EdgeDetector
from sudoku_digitalisation.features.sudoku_splitter import SudokuSplitter


class SudokuPreprocessor:
    """
    Handles full preprocessing tasks for single objects.
    """    
    def __init__(self,
                 # ADD EDGE DETECTOR ATTRIBUTES WHEN NECESSARY
                 clip_limit: int = 3,
                 output_size: int = 450) -> None:
        self.edge_detector = EdgeDetector()
        self.converter = ImageConverter(clip_limit)
        self.cropper = ImageCropper(output_size)

    def preprocess_image(self, image: Image.Image, keypoints: np.ndarray=None) -> Image.Image:
        clahe_img = self.converter.apply_clahe(image)
        bbox = self.edge_detector.get_bounding_box(clahe_img, keypoints)
        return self.cropper.crop_to_box(clahe_img, bbox)

    def preprocess_datapoint(self, dp: Dict[str, Any]) -> Dict[str, Any]:
        dp = dp.copy()
        dp['image'] = self.preprocess_image(dp['image'], dp['keypoints'])
        return dp
    
    def split_image(self, sudoku: Image.Image):
        return SudokuSplitter.split_image(sudoku)
    
    def split_datapoint(self, datapoint: Dict[str, Any]):
        return SudokuSplitter.split_datapoint(datapoint)
    
    def sudoku_preprocessing(self, sudoku: Union[Image.Image, Dict[str, Any]]) -> Tuple[Any, Any]:
        if isinstance(sudoku, Image.Image):
            preprocessed_img = self.preprocess_image(sudoku)
            digit_list = self.split_image(sudoku)
            return preprocessed_img, digit_list
        elif isinstance(sudoku, dict):
            preprocessed_dp = self.preprocess_datapoint(sudoku)
            labeled_digit_list = self.split_datapoint(sudoku)
            return preprocessed_dp, labeled_digit_list
        else:
            raise TypeError("Input must be a PIL.Image.Image or a dataset dictionary with an 'image' field.")
        

class DatasetPreprocessor(SudokuPreprocessor):
    """
    Handles full preprocessing tasks for datasets and splits.
    """
    def __init__(self,
                 # ADD EDGE DETECTOR ATTRIBTUES HERE
                 clip_limit: int = 3, 
                 output_size: int = 450) -> None:
        super().__init__(clip_limit, output_size)
        
    def split_preprocessing(self, split: str, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        preprocessed_list = []
        digits_list = []
        for datapoint in tqdm(dataset, desc=f"Preprocessing {split} split"):
            preprocessed_dp, digits_dict_list = self.sudoku_preprocessing(datapoint)
            preprocessed_list.append(preprocessed_dp)
            digits_list.extend(digits_dict_list)
        return Dataset.from_list(preprocessed_list), Dataset.from_list(digits_list)

    def dataset_preprocessing(self, dataset: DatasetDict):
        preprocessed_datasets = {}
        digits_datasets = {}
        for split in dataset:
            preprocessed_ds, digits_ds = self.split_preprocessing(split, dataset[split])
            preprocessed_datasets[split] = preprocessed_ds
            digits_datasets[split] = digits_ds
        return DatasetDict(preprocessed_datasets), DatasetDict(digits_datasets)

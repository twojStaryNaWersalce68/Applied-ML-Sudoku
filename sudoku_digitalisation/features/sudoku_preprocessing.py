import numpy as np
import cv2
from datasets import Dataset, DatasetDict
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import Dict, Tuple, Union, Any
from sudoku_digitalisation.features.edge_detector import EdgeDetector
from sudoku_digitalisation.features.sudoku_splitter import SudokuSplitter


class SudokuPreprocessor:
    """
    Handles full sudoku preprocessing tasks.
    """    
    def __init__(self,
                 # ADD EDGE DETECTOR ATTRIBUTES WHEN NECESSARY
                 clip_limit: int = 3,
                 output_size: int = 450) -> None:
        self.edge_detector = EdgeDetector()
        self.converter = ImageConverter(clip_limit)
        self.cropper = ImageCropper(output_size)

    def preprocess_sudoku(self, image: Image.Image, keypoints: np.ndarray=None) -> Image.Image:
        clahe_img = self.converter.apply_clahe(image)
        bbox = self.edge_detector.get_bounding_box(clahe_img, keypoints)
        cropped_img = self.cropper.crop_to_box(clahe_img, bbox)
        return cropped_img

    def preprocess_datapoint(self, dp: Dict[str, Any]) -> Dict[str, Any]:
        image = self.preprocess_sudoku(dp['image'], dp['keypoints'])
        dp['image'] = image
        return dp
    
    def split_sudoku(self, sudoku: Image.Image):
        return SudokuSplitter.split_image(sudoku)
    
    def split_datapoint(self, datapoint: Dict[str, Any]):
        return SudokuSplitter.split_datapoint(datapoint)
    
    def sudoku_preprocessing(self, sudoku: Union[Image.Image, Dict[str, Any]]) -> Tuple[Any, Any]:
        if isinstance(sudoku, Image.Image):
            preprocessed_img = self.preprocess_sudoku(sudoku)
            digit_list = self.split_sudoku(sudoku)
            return preprocessed_img, digit_list
        elif isinstance(sudoku, dict):
            preprocessed_dp = self.preprocess_datapoint(sudoku)
            labeled_digit_list = self.split_datapoint(sudoku)
            return preprocessed_dp, labeled_digit_list
        else:
            raise TypeError("Input must be a PIL.Image.Image or a dataset dictionary with an 'image' field.")
        
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


class ImageConverter:
    """
    Handles image color and contrast conversion tasks.
    """
    def __init__(self, clip_limit: int = 3) -> None:
        self.clip_limit = clip_limit

    def to_grayscale(self, image: Image.Image) -> Image.Image:
        return ImageOps.grayscale(image.convert("RGB"))

    def apply_clahe(self, image: Image.Image) -> Image.Image:
        image = self.to_grayscale(image)
        image_np = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit)
        clahe_np = clahe.apply(image_np)
        return Image.fromarray(clahe_np)


class ImageCropper:
    """
    Handles image position, perspective and cropping tasks.
    """
    def __init__(self, output_size: int) -> None:
        self._output_size = None
        self.desired_corner_points = None
        self.output_size = output_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @output_size.setter
    def output_size(self, value: int) -> None:
        self._output_size = value
        self.desired_corner_points = np.array([
            [0, 0],  # top left
            [0, value - 1],  # bottom left
            [value - 1, value - 1],  # bottom right
            [value - 1, 0]  # top right
        ], dtype=np.float32)

    def crop_to_box(self, image: Image.Image, bounding_box: np.ndarray) -> Image.Image:
        image_np = np.array(image)
        matrix = cv2.getPerspectiveTransform(bounding_box, self.desired_corner_points)
        warped = cv2.warpPerspective(image_np, matrix, (self.output_size, self.output_size))
        return Image.fromarray(warped)

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from PIL import Image
from typing import Dict, Tuple, Union, Any, List
from sudoku_digitalisation.features.image_operations import ImageConverter, ImageCropper
from sudoku_digitalisation.features.edge_detector import EdgeDetector
from sudoku_digitalisation.features.sudoku_splitter import SudokuSplitter
from sudoku_digitalisation.features.dataset_handler import DatasetHandler, load_sudoku_dataset


def get_preprocessor(
        clip_limit: int = 3,
        output_size: int = 450,
        is_preprocessed: bool = False,
        path: Union[str, None] = None
        ) -> 'DatasetPreprocessor':
    '''
    Gets the preprocessed datasets and makes it into a DatasetPreprocessor object,
    containing all the preprocessed datasets as well as the settings.
    '''
    print("Getting preprocessor dataset...")
    if is_preprocessed:
        handler = load_sudoku_dataset(path=path)
        preprocessor = DatasetPreprocessor(handler, clip_limit=clip_limit, output_size=output_size)
    else:
        handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True)
        preprocessor = DatasetPreprocessor(handler, clip_limit=clip_limit, output_size=output_size)
        preprocessor.dataset_preprocessing()
        preprocessor.handler.save_all_datasets()
    return preprocessor


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

    def convert_crop_image(self, image: Image.Image, keypoints: np.ndarray=None) -> Image.Image:
        bbox = self.edge_detector.get_bounding_box(image, keypoints)

        if bbox is None:
            print("Warning: Could not find a bounding box. Cannot crop image.")
            return None

        cropped_image = self.cropper.crop_to_box(image, bbox)
        final_processed_image = self.converter.apply_clahe(cropped_image)
        
        return final_processed_image

    def convert_crop_datapoint(self, dp: Dict[str, Any]) -> Dict[str, Any]:
        dp = dp.copy()
        dp['image'] = self.convert_crop_image(dp['image'], dp['keypoints'])
        return dp
    
    def sudoku_preprocessing(
            self,
            sudoku: Union[Image.Image, Dict[str, Any]]
            ) -> Tuple[Union[Image.Image, Dict[str, Any]], Union[List[Image.Image], Dataset]]:
        if isinstance(sudoku, Image.Image):
            preprocessed_img = self.convert_crop_image(sudoku)
            digit_list = SudokuSplitter.split_image(preprocessed_img)
            preprocessed_img.show()
            return preprocessed_img, digit_list
        elif isinstance(sudoku, dict):
            preprocessed_dp = self.convert_crop_datapoint(sudoku)
            labeled_digit_list = SudokuSplitter.split_datapoint(preprocessed_dp)
            return preprocessed_dp, labeled_digit_list
        else:
            raise TypeError("Input must be a PIL.Image.Image or a dataset dictionary with an 'image' field.")
        

class DatasetPreprocessor(SudokuPreprocessor):
    """
    Handles full preprocessing tasks for datasets and splits.
    """
    def __init__(self,
                 # ADD EDGE DETECTOR ATTRIBTUES HERE
                 handler: DatasetHandler,
                 clip_limit: int = 3, 
                 output_size: int = 450) -> None:
        self.handler = handler
        super().__init__(clip_limit, output_size)
        
    def split_preprocessing(self, split: str) -> Tuple[Dataset, Dataset]:
        preprocessed_list = []
        digits_list = []
        for datapoint in tqdm(self.handler.datasets['raw'][split], desc=f"Preprocessing {split} split"):
            preprocessed_dp, digits_dict_list = self.sudoku_preprocessing(datapoint)
            preprocessed_list.append(preprocessed_dp)
            digits_list.extend(digits_dict_list)
        return Dataset.from_list(preprocessed_list), Dataset.from_list(digits_list)

    def dataset_preprocessing(self) -> Tuple[DatasetDict, DatasetDict]:
        preprocessed_datasets = {}
        digits_datasets = {}
        for split in self.handler.datasets['raw']:
            preprocessed_ds, digits_ds = self.split_preprocessing(split)
            preprocessed_datasets[split] = preprocessed_ds
            digits_datasets[split] = digits_ds
        self.handler.datasets['preprocessed'] = DatasetDict(preprocessed_datasets)
        self.handler.datasets['digits'] = DatasetDict(digits_datasets)
        return self.handler.datasets['preprocessed'], self.handler.datasets['digits']

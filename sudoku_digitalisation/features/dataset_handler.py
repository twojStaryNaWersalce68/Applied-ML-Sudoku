import matplotlib.pyplot as plt
import os
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from typing import Tuple
from tqdm import tqdm
from sudoku_digitalisation.features.sudoku_preprocessing import SudokuPreprocessor


def load_sudoku_dataset(preprocessor, path=None, hugface=False) -> 'SudokuDatasetHandler':
    if hugface:
        dataset = load_dataset(path)
        return SudokuDatasetHandler(preprocessor, dataset=dataset)

    path = path if path is not None else r"sudoku_digitalisation\data"
    raw_path = os.path.join(path, "raw")
    prepro_path = os.path.join(path, "preprocessed")
    digits_path = os.path.join(path, "digits")

    dataset = load_from_disk(raw_path, keep_in_memory=True) if os.path.exists(raw_path) else None
    prepro_dataset = load_from_disk(prepro_path, keep_in_memory=True) if os.path.exists(prepro_path) else None
    digits_dataset = load_from_disk(digits_path, keep_in_memory=True) if os.path.exists(digits_path) else None

    if any([dataset, prepro_dataset, digits_dataset]):
        return SudokuDatasetHandler(preprocessor, dataset, prepro_dataset, digits_dataset)
    raise FileNotFoundError("No dataset found to load.")


class SudokuDatasetHandler:
    """Handles dataset managing."""

    def __init__(self,
                 preprocessor: SudokuPreprocessor,
                 dataset: DatasetDict=None,
                 preprocessed_dataset: DatasetDict=None,
                 digits_dataset: DatasetDict = None,
                 save_path: str=None) -> None:
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.preprocessed_dataset = preprocessed_dataset
        self.digits_dataset = digits_dataset
        self.save_path = save_path if save_path is not None else r"sudoku_digitalisation\data"

    def save(self) -> None:
        if self.dataset:
            raw_path = os.path.join(self.save_path, "raw")
            self.dataset.save_to_disk(raw_path)
        if self.preprocessed_dataset:
            preprocessed_path = os.path.join(self.save_path, "preprocessed")
            self.preprocessed_dataset.save_to_disk(preprocessed_path)
        if self.digits_dataset:
            digits_path = os.path.join(self.save_path, "digits")
            self.digits_dataset.save_to_disk(digits_path)

    def preprocess_dataset(self) -> None:
        pre, digits = self.preprocessor.dataset_preprocessing(self.dataset)
        self.preprocessed_dataset, self.digits_dataset = pre, digits

    def show_raw_image(self, split: str, index: int):
        if self.dataset is None:
            raise RuntimeError("Raw dataset not loaded.")
        image = self.dataset[split][index]['image']
        plt.imshow(image, cmap='gray')
        plt.title("Raw Image")
        plt.show()

    def show_preprocessed_image(self, split: str, index: int):
        if self.preprocessed_dataset is None:
            raise RuntimeError("Preprocessed dataset not loaded.")
        image = self.preprocessed_dataset[split][index]['image']
        plt.imshow(image, cmap='gray')
        plt.title("Preprocessed Image")
        plt.show()

    def show_digits_images(self, split: str, index: int):
        if self.digits_dataset is None:
            raise RuntimeError("Digits dataset not loaded.")
        fig, axes = plt.subplots(9, 9, figsize=(8, 8))
        for i in range(81):
            ax = axes[i // 9, i % 9]
            image = self.digits_dataset[split][index * 81 + i]['image']
            ax.imshow(image, cmap='gray')
            ax.axis('off')
        plt.suptitle(f"Digits from sudoku index {index}")
        plt.show()


#####################
### USAGE EXAMPLE ###
#####################

# If you get a ModuleNotFoundError, run:
# python -m sudoku_digitalisation.features.dataset_handler

if __name__ == '__main__':
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=450)

    # handler = load_sudoku_dataset(preprocessor, "Lexski/sudoku-image-recognition", hugface=True)
    # handler.save()

    handler = load_sudoku_dataset(preprocessor)

    handler.preprocess_dataset()
    handler.save()

    split = 'train'
    index = 29
    handler.show_raw_image(split, index)
    handler.show_preprocessed_image(split, index)
    handler.show_digits_images(split, index)

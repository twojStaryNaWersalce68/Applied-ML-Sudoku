import matplotlib.pyplot as plt
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from sudoku_digitalisation.features.sudoku_preprocessing import SudokuPreprocessor
from sudoku_digitalisation.features.edge_detector import EdgeDetector
from sudoku_digitalisation.features.sudoku_splitter import SudokuSplitter


class SudokuDatasetHandler:
    """Handles dataset loading, saving, and preprocessing."""

    def __init__(self,
                 preprocessor: SudokuPreprocessor,
                 dataset: DatasetDict=None,
                 preprocessed_dataset: DatasetDict=None,
                # digits_dataset = None,
                save_path: str=None) -> None:
        if save_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(os.path.dirname(current_dir), "data")
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.preprocessed_dataset = preprocessed_dataset
        # self.digits_dataset = digits_dataset
        self.save_path = save_path

    @staticmethod
    def load_data(preprocessor: SudokuPreprocessor,
                  path: str=None,
                  hugface: bool = False) -> 'SudokuDatasetHandler':
        if hugface:
            return SudokuDatasetHandler(preprocessor, dataset=load_dataset(path))
        elif path == None:
            current = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(os.path.dirname(current), "data")
        raw_path = os.path.join(path, "raw")
        prepro_path = os.path.join(path, "preprocessed")
        # digits_path = os.path.join(path, "digits")
        dataset = load_from_disk(raw_path, keep_in_memory=True) if os.path.exists(raw_path) else None
        prepro_dataset = load_from_disk(prepro_path, keep_in_memory=True) if os.path.exists(prepro_path) else None
        # digits_dataset = load_from_disk(digits_path, keep_in_memory=True) if os.path.exists(digits_path) else None
        if dataset is not None or prepro_dataset is not None: # or digits_dataset is not None
            return SudokuDatasetHandler(preprocessor, dataset, prepro_dataset) # digits_dataset
        else:
            return ValueError("No dataset found to load.")

    def save(self) -> None:
        if self.dataset:
            raw_path = os.path.join(self.save_path, "raw")
            self.dataset.save_to_disk(raw_path)
        if self.preprocessed_dataset:
            preprocessed_path = os.path.join(self.save_path, "preprocessed")
            self.preprocessed_dataset.save_to_disk(preprocessed_path)
        # if self.digit_dataset:
            # digits_path = os.path.join(self.save_path, "digits")
            # self.digits_dataset.save_to_disk(digits_path)

    def preprocess_dataset(self) -> DatasetDict:
        if self.dataset is None:
            raise RuntimeError("No dataset loaded for preprocessing.")
        ds = DatasetDict({split: dataset for split, dataset in self.dataset.items()})
        for split in ds:
            ds[split] = ds[split].map(self.preprocessor.preprocess_ds, num_proc=1)
        self.preprocessed_dataset = ds
        return ds

    def show_images(self, split: str, index: int) -> None:
        if self.dataset is not None:
            image = self.dataset[split][index]['image']
            plt.imshow(image, cmap="gray")
            plt.show()
        if self.preprocessed_dataset is not None:
            image = self.preprocessed_dataset[split][index]['image']
            plt.imshow(image, cmap="gray")
            plt.show()
        # if self.digits_dataset is not None:
        #     fig, axes = plt.subplots(9, 9, figsize=(8, 8))
        #     for i in range(81):
        #         ax = axes[i // 9, i % 9]
        #         image = self.digits_dataset[split][index + i]['image']
        #         ax.imshow(image, cmap="gray")
        #         ax.axis("off")
        #     plt.tight_layout()
        #     plt.show()


#####################
### USAGE EXAMPLE ###
#####################

# If you get a ModuleNotFoundError, run:
# python -m sudoku_digitalisation.features.dataset_handler

if __name__ == '__main__':
    edge_detector = EdgeDetector()
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=450)

    # handler = SudokuDatasetHandler.load_data(preprocessor, "Lexski/sudoku-image-recognition", hugface=True)
    # handler.save()

    handler = SudokuDatasetHandler.load_data(preprocessor)

    # handler.preprocess_dataset()
    # handler.save()

    handler.show_images('train', 29)

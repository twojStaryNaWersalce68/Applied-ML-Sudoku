import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from sudoku_digitalisation.features.image_operations import save_image


def load_sudoku_dataset(path=None, hugface=False) -> 'DatasetHandler':
    """Loads the dataset either from huggingface or from a local save."""
    if hugface:
        dataset = load_dataset(path)
        return DatasetHandler(dataset=dataset)

    if path is None:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, '..', 'data', 'datasets')
        print(path)
    raw_path = os.path.join(path, "raw")
    prepro_path = os.path.join(path, "preprocessed")
    digits_path = os.path.join(path, "digits")

    dataset = load_from_disk(raw_path, keep_in_memory=True) if os.path.exists(raw_path) else None
    prepro_dataset = load_from_disk(prepro_path, keep_in_memory=True) if os.path.exists(prepro_path) else None
    digits_dataset = load_from_disk(digits_path, keep_in_memory=True) if os.path.exists(digits_path) else None

    if any([dataset, prepro_dataset, digits_dataset]):
        return DatasetHandler(dataset, prepro_dataset, digits_dataset)
    raise FileNotFoundError("No dataset found to load.")


class DatasetHandler:
    """Handles dataset managing."""

    def __init__(self,
                 dataset: DatasetDict=None,
                 preprocessed_dataset: DatasetDict=None,
                 digits_dataset: DatasetDict = None,
                 save_path: str=None) -> None:
        self.datasets = {
            "raw": dataset,
            "preprocessed": preprocessed_dataset,
            "digits": digits_dataset
        }
        if save_path is None:
            base_dir = os.path.dirname(__file__)
            save_path = os.path.join(base_dir, '..', 'data', 'datasets')
        self.save_path = save_path

    def save_dataset(self, type: str, path: str = None) -> None:
        """Saves one dataset locally based on the specified type (raw, preprocessed, split into digits)."""
        if path is not None:
            self.save_path = path
        dataset = self.datasets[type]
        if dataset:
                dataset.save_to_disk(os.path.join(self.save_path, type))

    def save_all_datasets(self, path: str = None) -> None:
        """Saves all types (raw, preprocessed, split into digits) of datasets locally."""
        if path is not None:
            self.save_path = path
        for name in self.datasets:
            self.save_dataset(name, path)

    def save_split_png(self, type: str, split: str) -> None:
        """
        Saves the images of a given type (raw, preprocessed, split into digits) of dataset
        for a given split (test, train, validation).
        """
        dataset = self.datasets[type]
        for idx, datapoint in enumerate(tqdm(dataset[split], desc=f"Saving {split} split as png")):
            filename = f"{split}_{idx:05}.png"
            save_image(datapoint["image"], os.path.join(type, split), filename)

    def save_dataset_png(self, type: str) -> None:
        """
        Saves the images of a given type (raw, preprocessed, split into digits) of dataset
        for each split (test, train, validation).
        """
        dataset = self.datasets[type]
        for split in dataset:
            self.save_split_png(type, split)

    def show_image(self, type: str, split: str, index: int) -> None:
        """
        Shows an image at a given index,
        of a given type (raw, preprocessed, split into digits) of dataset,
        for a given split (test, train, validation).
        """
        dataset = self.datasets[type]
        if dataset is None:
            raise RuntimeError(f"{type} dataset not loaded.")
        image = dataset[split][index]['image']
        plt.imshow(image, cmap='gray')
        plt.show()

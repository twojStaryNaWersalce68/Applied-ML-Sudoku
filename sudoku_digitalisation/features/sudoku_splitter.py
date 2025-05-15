import numpy as np
from PIL import Image
from typing import List


class SudokuSplitter:
    @staticmethod
    def split_image(image: Image.Image) -> List[Image.Image]:
        """
        Splits the image into 81 cells.
        """
        image = np.array(image)
        cells = []
        cell_size = (image.shape[0] // 9, image.shape[1] // 9)
        for i in range(9):
            for j in range(9):
                x_start = i * cell_size[0]
                x_end = (i + 1) * cell_size[0]
                y_start = j * cell_size[1]
                y_end = (j + 1) * cell_size[1]
                cell = image[x_start:x_end, y_start:y_end]
                cells.append(Image.fromarray(cell))
        return cells
    
    @staticmethod
    def change_label(label: np.ndarray) -> int:
        """Changes the label of a single cell from binary representation to the simple integer."""
        label[0] = abs(label[0] - 1)  # changes 0 to 1 and 1 to 0, such that below if the cell is unsolved it returns 0
        for idx, item in enumerate(label):
            if item == 1:
                return idx
        return 0

    @staticmethod
    def split_labels(labels: np.ndarray) -> List[int]:
        """Splits the labels into a list of 81 labels."""
        new_labels = []
        for i in range(9):
            for j in range(9):
                new_label = SudokuSplitter.change_label(labels[i][j])
                new_labels.append(new_label)
        return new_labels

    @staticmethod
    def create_digit_ds(ds):
        """Sets up the datasets which we use for training."""
        new_ds = []
        for idx, row in enumerate(ds):
            digit_images = SudokuSplitter.split_image(row['image'])
            digit_labels = SudokuSplitter.split_labels(row['cells'])
            for j in range(len(digit_labels)):
                new_ds.append({"digit_img": digit_images[j], "label": digit_labels[j], "index": idx})
        return new_ds
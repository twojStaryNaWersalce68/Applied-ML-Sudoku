import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List
from datasets import Dataset


class SudokuPreprocessor:
    """Handles sudoku-specific preprocessing tasks."""

    def __init__(self, output_size: int=450) -> None:
        self.output_size = output_size

    def crop_to_bounding_box(
            self,
            image: Image.Image,
            keypoints: np.ndarray         
            ) -> Image.Image:

        bounding_box = np.array([
            [keypoints[0], keypoints[1]],  # top left
            [keypoints[2], keypoints[3]],  # bottom left
            [keypoints[4], keypoints[5]],  # bottom right
            [keypoints[6], keypoints[7]]   # top right
        ], dtype=np.float32)

        desired_corner_points = np.array([
            [0, 0],  # top left
            [0, self.output_size - 1],  # bottom left
            [self.output_size - 1, self.output_size - 1],  # bottom right
            [self.output_size - 1, 0]  # top right
        ], dtype=np.float32)

        image = np.array(image)
        pers_transform_matrix = cv2.getPerspectiveTransform(bounding_box, desired_corner_points)
        warped_image = cv2.warpPerspective(image, pers_transform_matrix, self.output_size)

        return Image.fromarray(warped_image)


    def hough_transform(ds):
        """
        Applies Hough transform to the image.
        This is done using cv2.HoughLinesP.
        """
        image = np.array(ds['image'])
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=500, maxLineGap=100)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        ds['image'] = Image.fromarray(image)
        return ds
    
    # def split_datapoint(self, ds) -> Dataset:

    def split_sudoku(self, sudoku: Image.Image) -> List[Image.Image]:
        """Splits the image into 81 cells."""
        image = np.array(sudoku)
        cells = []
        cell_size = image.shape[0] // 9
        for i in range(9):
            for j in range(9):
                x_start = i * cell_size
                x_end = (i + 1) * cell_size
                y_start = j * cell_size
                y_end = (j + 1) * cell_size
                cell = image[y_start:y_end, x_start:x_end]
                cells.append(Image.fromarray(cell))
        return cells
    
    # def split_label(self, label):
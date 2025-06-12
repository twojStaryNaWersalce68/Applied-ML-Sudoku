import numpy as np
from PIL import Image
from typing import List
import cv2
from typing import Tuple
from tensorflow.python.ops.gen_array_ops import lower_bound


class CellSplitter:
    @staticmethod
    def get_candidate_digits(cell_sample: Image.Image) -> List[Image.Image]:
        """ Retrieves the candidate digits from the given cell sample. """
        cell = np.array(cell_sample)
        background_colour, thresh_setting = CellSplitter.determine_mode(cell)
        processed_cell = CellSplitter.replace_cell_frame(cell, background_colour)
        binary_cell = CellSplitter.convert_to_binary(processed_cell, thresh_setting)
        bounding_boxes = CellSplitter.find_bounding_boxes(binary_cell)
        candidate_digits = CellSplitter.crop_image(cell, bounding_boxes)
        return candidate_digits

    @staticmethod
    def determine_mode(image: np.array) -> Tuple[List[int], int]:
        """
        Determines the mean colour of the pixels in the cell and returns this,
        as well as the binary threshold setting, which depends on whether the cell is
        black writing on white or white writing on black.
        """
        threshold = 255 / 2
        mean_colour = image.mean()
        if mean_colour <= threshold:
            return (mean_colour, mean_colour, mean_colour), cv2.THRESH_BINARY_INV
        return (mean_colour, mean_colour, mean_colour), cv2.THRESH_BINARY

    @staticmethod
    def replace_cell_frame(image: np.ndarray, bg_colour: List[int]) -> np.ndarray:
        """
        Replaces the outer 2 pixels of the cell with the mean colour of the cell.
        This is necessary to detect digits, which are connected to the edges of the image.
        """
        frame_w = 2
        h, w = image.shape
        image = image[frame_w:(h-frame_w), frame_w:(w-frame_w)]
        image = cv2.copyMakeBorder(image, frame_w, frame_w, frame_w, frame_w, cv2.BORDER_CONSTANT, value=bg_colour)
        return image

    @staticmethod
    def convert_to_binary(image: np.ndarray, thresh_setting: int) -> np.ndarray:
        """Converts the given image to a binary image."""
        _, binary_image = cv2.threshold(image, 0, 255, thresh_setting + cv2.THRESH_OTSU)
        return binary_image

    @staticmethod
    def find_bounding_boxes(image: np.ndarray) -> List[List[int]]:
        """
        Determines the bounding boxes from the given image.
        To filter out noise and the outer corners of the image, a min and max are set for what is considered.
        """
        min_area = 20
        max_area = 300

        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append([x, y, w, h])

                # below draws the bounding boxes into the image, if we want to visualize
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return bounding_boxes

    @staticmethod
    def find_cropping_dimensions(b_box: List[int], img_shape: List[int]) -> tuple[int, int, int]:
        """
        For the final images to have the right shape, the bounding boxes need to be square.
        They also need to be within the image boundaries.
        This function makes sure the dimensions used for cropping the images fulfil that criteria.
        """
        x, y, w, h = b_box
        centre_x = x + (w / 2)
        centre_y = y + (h / 2)
        size = max(w, h)  # the dimension of the square is determined by the larger side of the rectangle (bbox)
        # determines new (x, y) based on the new dimensions
        x = max(0, int(centre_x - size / 2))
        y = max(0, int(centre_y - size / 2))
        # determines the max dimensions from (x, y) based on the image shape
        img_h, img_w = img_shape
        max_h = img_h - y
        max_w = img_w - x
        size = min(size, max_h, max_w)
        return x, y, size

    @staticmethod
    def crop_image(image: np.ndarray, b_boxes: List[List[int]]) -> List[Image.Image]:
        """Crops the image to the individual digits based on the given bounding boxes."""
        candidate_digits = []
        for b_box in b_boxes:
            x, y, size = CellSplitter.find_cropping_dimensions(b_box, image.shape)
            cropped_image = image[y:y + size, x:x + size]
            cropped_image = cv2.resize(cropped_image, (50, 50))  # Add this to reshape_image in CNN later
            candidate_digits.append(Image.fromarray(cropped_image))
        return candidate_digits



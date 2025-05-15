import numpy as np
import cv2
from PIL import Image, ImageOps
from datasets import Dataset


class SudokuPreprocessor:
    """Handles image preprocessing tasks."""
    
    def __init__(
            self,
            edge_detector: 'EdgeDetector',
            clip_limit: int=3,
            output_size: int=450
            ) -> None:

        self.edge_detector = edge_detector
        self.clip_limit = clip_limit
        self._output_size = None
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

    def _convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        rgb_image = image.convert("RGB")
        return ImageOps.grayscale(rgb_image)

    def _apply_clahe(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit)
        clahe_np = clahe.apply(image_np)
        return Image.fromarray(clahe_np)
    
    def crop_to_bounding_box(
            self,
            image: Image.Image,
            bounding_box: np.ndarray         
            ) -> Image.Image:
        image = np.array(image)
        pers_transform_matrix = cv2.getPerspectiveTransform(bounding_box, self.desired_corner_points)
        warped_image = cv2.warpPerspective(image, pers_transform_matrix, (self.output_size, self.output_size))
        return Image.fromarray(warped_image)
    
    def preprocess_image(self, image: Image.Image, keypoints: np.ndarray=None) -> Image.Image:
        grayscale = self._convert_to_grayscale(image)
        clahe_img = self._apply_clahe(grayscale)
        # bbox = self.edge_detector.get_bounding_box(clahe_img, keypoints)
        # cropped_img = self.crop_to_bounding_box(clahe_img, bbox)
        return clahe_img

    def preprocess_ds(self, ds: Dataset) -> Dataset:
        image = self.preprocess_image(ds['image'], ds['keypoints'])
        ds['image'] = image
        return ds
    

class EdgeDetector:

    def __init__(self) -> None:
        # EDGE DETECTION SETTINGS
        pass

    def get_bounding_box(
            self,
            image: Image.Image,
            keypoints: np.ndarray=None
            ) -> np.ndarray:
        if keypoints != None:
            bounding_box = np.array([
                [keypoints[0], keypoints[1]],  # top left
                [keypoints[2], keypoints[3]],  # bottom left
                [keypoints[4], keypoints[5]],  # bottom right
                [keypoints[6], keypoints[7]]   # top right
            ], dtype=np.float32)
            return bounding_box
        else:
            # PERFORM EDGE DETECTION TO GET BOUNDING BOX
            pass


class SudokuSplitter:
    pass

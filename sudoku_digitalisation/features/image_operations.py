import numpy as np
import cv2
from PIL import Image, ImageOps


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
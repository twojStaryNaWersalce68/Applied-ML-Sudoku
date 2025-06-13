import numpy as np
import cv2
import os
from PIL import Image, ImageOps


def save_image(image: Image.Image, type: str, name: str) -> None:
    """Saves one image."""
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, '..', 'data', 'viewable_images', type, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

class ImageConverter:
    """
    Handles image color and contrast conversion tasks.
    """
    def __init__(self, clip_limit: int = 3) -> None:
        self.clip_limit = clip_limit

    def to_grayscale(self, image: Image.Image) -> Image.Image:
        """Converts image to grayscale."""
        return ImageOps.grayscale(image.convert("RGB"))

    def apply_clahe(self, image: Image.Image) -> Image.Image:
        """Enhances local contrast of image."""
        image = self.to_grayscale(image)
        image_np = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit)
        clahe_np = clahe.apply(image_np)
        return Image.fromarray(clahe_np)
    
    def apply_gaussian_blur(self, image: Image.Image, kernel_size: tuple = (5, 5)) -> Image.Image:
        """Applies a Gaussian Blur."""
        image_np = np.array(image)
        blurred_np = cv2.GaussianBlur(image_np, kernel_size, 0)
        return Image.fromarray(blurred_np)

    def apply_adaptive_threshold(self, image: Image.Image) -> Image.Image:
        """
        Applies an adaptive threshold.
        This has to be performed on a greyscale image.
        """
        grayscale_image = self.to_grayscale(image)
        image_np = np.array(grayscale_image)
        
        binary_np = cv2.adaptiveThreshold(
            image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return Image.fromarray(binary_np)


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
        """This set's the desired output size used for cropping."""
        self._output_size = value
        self.desired_corner_points = np.array([
            [0, 0],  # top left
            [0, value - 1],  # bottom left
            [value - 1, value - 1],  # bottom right
            [value - 1, 0]  # top right
        ], dtype=np.float32)

    def crop_to_box(self, image: Image.Image, bounding_box: np.ndarray) -> Image.Image:
        """Crops the image based on bounding box to the desired output size."""
        image_np = np.array(image)
        matrix = cv2.getPerspectiveTransform(bounding_box, self.desired_corner_points)
        warped = cv2.warpPerspective(image_np, matrix, (self.output_size, self.output_size))
        return Image.fromarray(warped)
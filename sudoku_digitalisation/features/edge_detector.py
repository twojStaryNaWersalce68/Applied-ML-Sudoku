from PIL import Image
import cv2
import numpy as np
from sudoku_digitalisation.features.image_operations import ImageConverter


class EdgeDetector:
    """
    Handles the edge detection algorithm to find the bounding boxes around the sudokus without given keypoints.
    """

    def __init__(self) -> None:
        pass

    def _known_keypoints_bb(self, keypoints) -> np.ndarray | None:
        """Reformats the bounding box to the correct format for the cropping."""
        bounding_box = np.array([
                [keypoints[0], keypoints[1]],  # top left
                [keypoints[2], keypoints[3]],  # bottom left
                [keypoints[4], keypoints[5]],  # bottom right
                [keypoints[6], keypoints[7]]   # top right
            ], dtype=np.float32)
        return bounding_box

    def get_bounding_box(
            self,
            image: Image.Image,
            keypoints: np.ndarray=None
            ) -> np.ndarray:
        """Retrieves the bounding box around the sudokus with given keypoints or by finding keypoints."""
        if keypoints != None:
            return self._known_keypoints_bb(keypoints)
        else:
            return self._get_keypoints_bb(image)
    
    def _get_keypoints_bb(self, image) -> np.ndarray | None:
        """
        Gets the bounding box keypoints using edge detection.
        """ 
        # Works for both PIL and NumPY array
        if isinstance(image, Image.Image):
            original_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            original_image = image

        converter = ImageConverter()
        original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        grayscale_pil = converter.to_grayscale(original_image_pil)
        blurred_pil = converter.apply_gaussian_blur(grayscale_pil)
        binary_pil = converter.apply_adaptive_threshold(blurred_pil)
        
        binary_np = np.array(binary_pil)
        
        grid_contour = self._find_grid_contour(binary_np)
        
        if grid_contour is None:
            return None

        corners = self._order_corners(grid_contour)

        return corners
    
    def _find_grid_contour(self, processed_image: np.ndarray) -> np.ndarray | None:
        """Finds the largest contour that is a sudoku"""
        h, w = processed_image.shape[:2]
        total_area = h * w
        
        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if (len(approx) == 4 and 
                area > total_area * 0.1 and
                area > max_area):
                largest_contour = approx
                max_area = area
        return largest_contour

    def _order_corners(self, contour: np.ndarray) -> np.ndarray:
        """Orders the corners: tl, tr, br, bl"""
        points = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)] # Top-left
        rect[2] = points[np.argmax(s)] # Bottom-right
        diff = np.diff(points, axis=1)
        rect[3] = points[np.argmin(diff)] # Top-right
        rect[1] = points[np.argmax(diff)] # Bottom-left
        return rect
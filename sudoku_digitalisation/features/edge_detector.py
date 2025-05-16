import numpy as np
import cv2
from PIL import Image


class EdgeDetector:

    def __init__(self, t_lower, t_upper) -> None:
        self.t_lower = t_lower
        self.t_upper = t_upper

    def _keypoints_bb(self, keypoints: np.ndarray) -> np.ndarray:
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
        if keypoints != None:
            return self._keypoints_bb(keypoints)
        else:
            edges = self.find_contours(image)
            return self._keypoints_bb(edges)
        
    def canny(self, image: Image.Image) -> np.ndarray:
        np_img = np.array(image)
        edged_img = cv2.Canny(np_img, self.t_lower, self.t_upper)
        Image.fromarray(edged_img).show()
        return edged_img
    
    def find_contours(self, image: Image.Image):
        canny_img = self.canny(image)
        contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        np_img = np.array(image)
        cv2.drawContours(np_img, contours, -1, (0, 255, 0), 3)
        Image.fromarray(np_img).show()

import numpy as np
import os
import joblib
from PIL import Image
from sklearn import svm
from typing import List, Union, Tuple
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


class SVM():
    def __init__(self, input_shape: Tuple[int, int], verbose: bool = False) -> None:
        '''
        Initialise svm as a one versus all SVM
        '''
        self.n_features = input_shape[0] * input_shape[1]
        self.model = svm.LinearSVC(verbose=verbose)

    def _reshape_image_SVM(self, img: Image.Image) -> np.ndarray:
        '''
        Reshapes single image to match input shape
        '''
        img_array = np.array(img)
        normalized_array = img_array.astype(np.float32) / 255.0
        return normalized_array.flatten()

    def _reshape_data_SVM(self, image_list: List[Image.Image]) -> np.ndarray:
        '''
        Reshapes a list of images to match input shape
        '''
        reshaped_data = np.zeros((len(image_list), self.n_features), dtype=np.float32)
        for i, img in enumerate(image_list):
            reshaped_data[i] = self._reshape_image_SVM(img)
        return reshaped_data

    def train(self, X_train: List[Image.Image], y_train: List[int]) -> None:
        '''
        Trains svm on X_train matrix and y_train vector
        '''
        if len(X_train) > 5000:
            X_train = X_train[:5000]
            y_train = y_train[:5000]
        X_train = self._reshape_data_SVM(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train)

    def predict(self, input: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        '''
        Predict value(s) using the trained SVM
        '''
        if isinstance(input, Image.Image):
            input = self._reshape_image_SVM(input)
        else:
            input = self._reshape_data_SVM(input)
        return self.model.predict(input)

    def evaluate(self, X_test: List[Image.Image], y_test: List[int]) -> None:
        '''
        Evaluates the svm using the passed test data
        '''
        CELL_NUM = 81
        y_test = np.array(y_test)
        y_pred = self.predict(X_test)

        # Overall accuracy
        test_accuracy = accuracy_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Per class precisionm recall and F1
        classes = [str(i) for i in range(10)]
        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

        # Sudoku accuracy
        num_sudokus = len(y_test)//CELL_NUM
        correct_sudokus = 0
        for i in range(num_sudokus):
            start = i * CELL_NUM
            end = start + CELL_NUM
            if np.array_equal(y_pred[start:end], y_test[start:end]):
                correct_sudokus += 1
        correct_percent = correct_sudokus/num_sudokus

        return cm, {
            "test accuracy": test_accuracy,
            "sudoku accuracy w/o ED": correct_percent,
            "precision macro": report["macro avg"]["precision"],
            "recall macro": report["macro avg"]["recall"],
            "f1 macro": report["macro avg"]["f1-score"]
        }, None

    def save(self, name: str, path: str=None) -> None:
        """Saves the model."""
        if path is None:
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, "saved", "svm")
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{name}.joblib")
        joblib.dump(self.model, save_path)

    def load(self, name: str, path: str=None) -> None:
        """Loads the model from a local save."""
        if path is None:
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, "saved", "svm")
        load_path = os.path.join(path, f"{name}.joblib")
        self.model = joblib.load(load_path)

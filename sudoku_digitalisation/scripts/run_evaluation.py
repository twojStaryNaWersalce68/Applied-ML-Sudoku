import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple, List
from PIL import Image

from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.scripts.run_comparison import show_cm
from sudoku_digitalisation.scripts.run_prediction import make_prediction
from sudoku_digitalisation.features.sudoku_splitter import SudokuSplitter


def full_evaluation(
        cnn: CNN,
        preprocessor: DatasetPreprocessor        
        ) -> None:
    '''
    Compares two one-hot encoded sudokus, returns true if they are the same.
    '''
    solution_correct = 0
    candidate_correct = 0
    X_test = preprocessor.handler.datasets['raw']['test']['image']
    y_test = preprocessor.handler.datasets['raw']['test']['cells']
    for i, image in enumerate(X_test):
        large_digits, binary_pred = make_prediction(image, preprocessor, cnn, binary=True)
        if large_digits == SudokuSplitter.split_labels(y_test[i]):
            solution_correct += 1
        if np.array_equal(binary_pred, y_test[i]):
            candidate_correct += 1
    solution_percentage = solution_correct / len(X_test) * 100
    candidate_percentage = candidate_correct / len(X_test) * 100
    print(f"Percentage of fully correct sudokus without candidate digits: {solution_percentage}%")
    print(f"Percentage of fully correct sudokus with candidate digits: {candidate_percentage}%")

def compare_bbox(
        true_bbox: np.ndarray,
        pred_bbox: np.ndarray,
        tolerance: int = 20
        ) -> bool:
    '''
    Compares two bounding boxes, returns true if the mean distance between
    all points is lower than "tolerance".
    '''
    distances = np.linalg.norm(true_bbox - pred_bbox, axis=-1)
    mean_distance = np.mean(distances)
    return mean_distance <= tolerance

def evaluate_edge_detection(preprocessor: DatasetPreprocessor) -> None:
    '''
    Prints the edge detection accuracy, tested on the training set.
    '''
    prediction_correct = []
    image_dataset = preprocessor.handler.datasets['raw']['train']['image']
    label_dataset = preprocessor.handler.datasets['raw']['train']['keypoints']
    for i in range(len(image_dataset)):
        true_bbox = preprocessor.edge_detector.get_bounding_box(
            image_dataset[i], label_dataset[i]
            )
        pred_bbox = preprocessor.edge_detector.get_bounding_box(
            image_dataset[i]
            )
        if pred_bbox is None:
            prediction_correct.append(False)
        else:
            prediction_correct.append(compare_bbox(true_bbox, pred_bbox))
    accuracy = sum(prediction_correct) / len(prediction_correct) * 100
    print(f"Edge detection accuracy: {accuracy}%")

def get_test(preprocessor: DatasetPreprocessor) -> Tuple[List[Image.Image], List[int]]:
    '''
    Get the test set from the preprocessor.
    '''
    digit_dataset = preprocessor.handler.datasets['digits']

    X_test = digit_dataset['test']['image']
    y_test = digit_dataset['test']['label']

    return X_test, y_test

def evaluate_model(
        model: Union[CNN, SVM],
        preprocessor: DatasetPreprocessor,
        trained: bool
        ) -> None:
    '''
    Evaluates the model's accuracy, precision, recall and F1
    '''
    X_test, y_test = get_test(preprocessor)

    cm, info, history = model.evaluate(X_test, y_test)

    # Confusion matrix
    show_cm(cm)

    # Per class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracies):
        print(f"Accuracy for digit {i}: {acc:.4f}")
    print(f"Test set accuracy: {info['test accuracy']:.4f}")

    # Per class precision, recall and F1
    print(f"Test set precision: {info['precision macro']:.4f}")
    print(f"Test set recall: {info['recall macro']:.4f}")
    print(f"Test set F1 score: {info['f1 macro']:.4f}")

    # accuracy on sudokus
    print(f"Accuracy for full sudokus w/o edge detection: {info['sudoku accuracy w/o ED']}")

    if trained:
        # Accuracy plot over time
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training vs Validation Accuracy')
        plt.show()

        # Loss plot over time
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.scripts.run_comparison import show_cm

def compare_bbox(
        true_bbox: np.ndarray,
        pred_bbox: np.ndarray,
        tolerance: int = 20
        ) -> bool:
    distances = np.linalg.norm(true_bbox - pred_bbox, axis=-1)
    mean_distance = np.mean(distances)
    return mean_distance <= tolerance

def evaluate_edge_detection(preprocessor: DatasetPreprocessor) -> None:
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

def get_test(preprocessor: DatasetPreprocessor):
    '''
    Get the test set from the preprocessor.
    '''
    digit_dataset = preprocessor.handler.datasets['digits']

    X_test = digit_dataset['test']['image']
    y_test = digit_dataset['test']['label']

    return X_test, y_test


def get_binary_labels(preprocessor: DatasetPreprocessor) -> List[int]:
    """
    Get the binary labels of the test set from the preprocessor.
    """
    y_raw_dataset = preprocessor.handler.datasets['raw']['test']['cells']
    y_raw = []
    for sudoku in y_raw_dataset:
        for row in sudoku:
            for cell in row:
                y_raw.append(cell)
    return y_raw


def evaluate_model(model: Union[CNN, SVM], preprocessor: DatasetPreprocessor, trained: bool):
    '''
    Evaluates the model's accuracy, precision, recall and F1
    '''
    X_test, y_test = get_test(preprocessor)
    y_binary = get_binary_labels(preprocessor)

    cm, info, history = model.evaluate(X_test, y_test, y_binary)

    # Confusion matrix
    show_cm(cm)

    # Per class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracies):
        print(f"Accuracy for digit {i}: {acc:.4f}")
    print(f"Test set accuracy: {info['test accuracy']:.4f}")
    print(f"Test set accuracy with candidate digits: {info['test accuracy candidate']:.4f}")

    # Per class precision, recall and F1
    print(f"Test set precision: {info['precision macro']:.4f}")
    print(f"Test set recall: {info['recall macro']:.4f}")
    print(f"Test set F1 score: {info['f1 macro']:.4f}")

    # accuracy on sudokus
    print(f"Accuracy for fully correct sudokus: {info['sudoku accuracy']}")

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

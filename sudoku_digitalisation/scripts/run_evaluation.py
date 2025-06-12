import matplotlib.pyplot as plt
from typing import Union
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sklearn.metrics import ConfusionMatrixDisplay

def get_test(preprocessor: DatasetPreprocessor):
    '''
    Get the test set from the preprocessor.
    '''
    digit_dataset = preprocessor.handler.datasets['digits']

    X_test = digit_dataset['test']['image']
    y_test = digit_dataset['test']['label']

    return X_test, y_test


def get_binary_labels(preprocessor: DatasetPreprocessor):
    """
    Get the binary labels of the test set from the preprocessor.
    """
    y_raw_dataset = preprocessor.handler.datasets['raw']['test']['cells']
    return y_raw_dataset


def evaluate_model(model: Union[CNN, SVM], preprocessor: DatasetPreprocessor, trained: bool):
    '''
    Evaluates the model's accuracy, precision, recall and F1
    '''
    X_test, y_test = get_test(preprocessor)

    cm, info, history = model.evaluate(X_test, y_test)

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
    disp.plot(cmap='viridis', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

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

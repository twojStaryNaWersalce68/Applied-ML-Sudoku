import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, Any, List
from sudoku_digitalisation.models.CNN import CNN
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

def evaluate_model(cnn: CNN, preprocessor: DatasetPreprocessor, trained: bool):
    '''
    Evaluates the model's accuracy, precision, recall and F1
    '''
    X_test, y_test = get_test(preprocessor)

    cm, info, history = cnn.evaluate(X_test, y_test)

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
    disp.plot(cmap='viridis', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    # Per class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracies):
        print(f"Accuracy for Class {i}: {acc:.4f}")
    print(f"Accuracy for test set: {info['test accuracy']:.4f}")

    # Per class precision, recall and F1
    print(f"The class precision is: {info['precision macro']:.4f}")
    print(f"The recall is: {info['recall macro']:.4f}")
    print(f"The f1 score is: {info['f1 macro']:.4f}")

    # accuracy on sudokus
    print(f"accuracy of fully correct sudokus: {info['sudoku accuracy']}")

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

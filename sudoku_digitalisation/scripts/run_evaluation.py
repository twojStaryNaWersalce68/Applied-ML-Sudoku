import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from typing import Tuple, Dict, List, Union
from sklearn.metrics import ConfusionMatrixDisplay
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM


def get_test(preprocessor: DatasetPreprocessor):
    '''
    Get the test set from the preprocessor.
    '''
    digit_dataset = preprocessor.handler.datasets['digits']

    X_test = digit_dataset['test']['image']
    y_test = digit_dataset['test']['label']

    return X_test, y_test

def evaluate_model(model: Union[CNN, SVM], preprocessor: DatasetPreprocessor, trained: bool):
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

def kfold_evaluation(
        model_type: str,
        preprocessor: DatasetPreprocessor,
        k: int,
        val_split: float = 0.2
        ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Perform k-fold cross-validation and return mean and std of metrics.
    """
    kf = KFold(n_splits=k, shuffle=True)
    cm_total = None
    metrics_folds = []

    train = preprocessor.handler.datasets['digits']['train']
    val = preprocessor.handler.datasets['digits']['validation']
    test = preprocessor.handler.datasets['digits']['test']

    images = np.array(train['image'] + val['image'] + test['image'])
    labels = np.array(train['label'] + val['label'] + test['label'])

    dim = preprocessor.cropper.output_size // 9

    for train_index, test_index in kf.split(images):
        X_train_full, X_test = images[train_index], images[test_index]
        y_train_full, y_test = labels[train_index], labels[test_index]

        if model_type == 'cnn':
            val_size = int(len(X_train_full) * val_split)
            X_val = X_train_full[:val_size]
            y_val = y_train_full[:val_size]
            X_train = X_train_full[val_size:]
            y_train = y_train_full[val_size:]

            model = CNN(input_shape=(dim, dim, 1), num_classes=10)
            model.train(X_train, y_train, X_val, y_val)
        elif model_type == 'svm':
            model = SVM(input_shape=(dim, dim))
            model.train(X_train_full, y_train_full)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Expected 'cnn' or 'svm'.")

        cm, metrics, _ = model.evaluate(X_test, y_test)
        if cm_total is None:
            cm_total = cm
        else:
            cm_total += cm
        metrics_folds.append(metrics)

    return cm_total, compute_mean_std(metrics_folds)   


def compute_mean_std(results: List[Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    metric_names = results[0].keys()

    summary = {}
    for metric in metric_names:
        values = np.array([fold[metric] for fold in results])
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    return summary


def compare_cnn_svm(
        preprocessor: DatasetPreprocessor,
        k: int
        ) -> None:
    cnn_cm, cnn_metrics = kfold_evaluation('cnn', preprocessor, k)
    svm_cm, svm_metrics = kfold_evaluation('svm', preprocessor, k)

    print(f"{'Metric':<15} | {'CNN':^15} | {'SVM':^15}")
    print("-" * 50)
    for metric in cnn_metrics.keys():
        cnn_mean = cnn_metrics[metric]["mean"]
        cnn_std = cnn_metrics[metric]["std"]
        svm_mean = svm_metrics[metric]["mean"]
        svm_std = svm_metrics[metric]["std"]
        print(f"{metric:<15} | {cnn_mean:.4f} ± {cnn_std:.4f} | {svm_mean:.4f} ± {svm_std:.4f}")
    print()

    show_cm(cnn_cm)
    show_cm(svm_cm)
    

def show_cm(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

import numpy as np
from typing import Optional, Union, Tuple
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor


def get_model(
        model_type: Optional[str],
        model_name: Optional[str],
        preprocessor: DatasetPreprocessor
        ) -> Optional[Union[CNN, SVM]]:

    if model_type is None:
        return None

    dim = preprocessor.cropper.output_size // 9

    if model_type == 'cnn':
        print("Getting CNN...")
        cnn = CNN(input_shape=(dim, dim, 1), num_classes=10)
        if model_name is None:
            X_train, y_train, X_val, y_val = get_train_val(preprocessor)
            cnn.train(X_train, y_train, X_val, y_val)
        else:
            cnn.load(model_name)
        return cnn

    elif model_type == 'svm':
        print("Getting SVM...")
        svm = SVM(input_shape=(dim, dim))
        if model_name is None:
            X_train, y_train, _, _ = get_train_val(preprocessor)
            svm.train(X_train, y_train)
        else:
            svm.load(model_name)
        return svm

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'cnn', 'svm', or None.")


def get_train_val(preprocessor: DatasetPreprocessor):
    digit_dataset = preprocessor.handler.datasets['digits']

    X_train = digit_dataset['train']['image']
    y_train = digit_dataset['train']['label']

    X_val = digit_dataset['validation']['image']
    y_val = digit_dataset['validation']['label']

    return X_train, y_train, X_val, y_val
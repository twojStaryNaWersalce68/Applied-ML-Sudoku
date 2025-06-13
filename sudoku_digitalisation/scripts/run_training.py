from typing import Optional, Union, List
from PIL import Image
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor


def get_model(
        train_model: bool,
        model_type: str,
        model_name: str,
        preprocessor: DatasetPreprocessor
        ) -> Optional[Union[CNN, SVM]]:
    '''
    Loads or trains a CNN or SVM, depending on the parameters.
    '''

    dim = preprocessor.cropper.output_size // 9

    if model_type == 'cnn':
        cnn = CNN(input_shape=(dim, dim, 1), num_classes=10)
        if train_model:
            print("Training CNN...")
            X_train, y_train, X_val, y_val = get_train_val(preprocessor)
            cnn.train(X_train, y_train, X_val, y_val, verbose=1)
            cnn.save(model_name)
        else:
            print("Getting CNN...")
            cnn.load(model_name)
        return cnn
    elif model_type == 'svm':
        svm = SVM(input_shape=(dim, dim), verbose=True)
        if train_model:
            print("Training SVM...")
            X_train, y_train, _, _ = get_train_val(preprocessor)
            svm.train(X_train, y_train)
            svm.save(model_name)
        else:
            print("Getting SVM...")
            svm.load(model_name)
        return svm
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'cnn' or 'svm'.")


def get_train_val(preprocessor: DatasetPreprocessor) -> List[List[Image.Image], List[int], List[Image.Image], List[int]]:
    '''
    Get the train and validation splits from the preprocessor.
    '''
    digit_dataset = preprocessor.handler.datasets['digits']

    X_train = digit_dataset['train']['image']
    y_train = digit_dataset['train']['label']

    X_val = digit_dataset['validation']['image']
    y_val = digit_dataset['validation']['label']

    return X_train, y_train, X_val, y_val
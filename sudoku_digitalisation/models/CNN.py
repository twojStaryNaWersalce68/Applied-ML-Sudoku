import os
import keras
import keras_tuner as kt
import numpy as np
from PIL import Image
from typing import Tuple, List, Union
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
    )

CELL_NUM = 81

class CNN:
    def __init__(
            self,
            input_shape: Tuple[int, int, int], 
            num_classes: int
            ) -> None:
        '''
        Initialize CNN with given parameters
        '''
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self, hp) -> keras.models.Sequential:
        '''
        Build the model
        '''
        input = keras.Input(shape=self.input_shape)

        hp_filters1 = hp.Choice('conv_filters_1', values=[8, 16, 32])

        conv1A = keras.layers.Conv2D(filters=hp_filters1, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu')(input)
        maxpool1A = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1A)
        conv1B = keras.layers.Conv2D(filters=hp_filters1, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu')(maxpool1A)
        maxpool1B = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1B)
        dropout1 = keras.layers.Dropout(0.2)(maxpool1B)

        hp_filters2 = hp.Choice('conv_filters_2', values=[32, 64, 128])

        conv2A = keras.layers.Conv2D(filters=hp_filters2, kernel_size=(3, 3), activation='relu')(dropout1)
        maxpool2A = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2A)
        conv2B = keras.layers.BD(filters=hp_filters2, kernel_size=(3, 3), activation='relu')(maxpool2A)
        maxpool2B = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2B)
        dropout2 = keras.layers.Dropout(0.2)(maxpool2B)

        flatten = keras.layers.Flatten()(dropout2)
        
        hp_filters3 = hp.Choice('dense_units', values=[64, 128, 256])

        dense = keras.layers.Dense(units=hp_filters3, activation='relu')(flatten)
        dropout3 = keras.layers.Dropout(0.2)(dense)

        output = keras.layers.Dense(units=self.num_classes, activation='softmax')(dropout3)

        model = keras.Model(inputs=input, outputs=output)

        # Hyperparameters
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        return model
    
    def _reshape_image_CNN(self, img: Image.Image) -> np.ndarray:
        '''
        Reshapes single image to match input shape.
        '''
        img_array = np.array(img)
        normalized_array = img_array.astype(np.float32) / 255.0
        return normalized_array.reshape(self.input_shape[0], self.input_shape[1], 1)

    def _reshape_data_CNN(self, image_list: List[Image.Image]) -> np.ndarray:
        '''
        Reshapes a list of images to match input shape.
        '''
        reshaped_data = np.zeros((
            len(image_list), self.input_shape[0], self.input_shape[1], 1
            ), dtype=np.float32)
        for i, img in enumerate(image_list):
            reshaped_data[i] = self._reshape_image_CNN(img)
        return reshaped_data
    
    def train(
            self,
            X_train: List[Image.Image],
            y_train: List[int],
            X_val: List[Image.Image],
            y_val: List[int],
            verbose: int = 1
            ) -> None:
        '''
        Train CNN using the training and validation data
        '''
        X_train = self._reshape_data_CNN(X_train)
        y_train = keras.utils.to_categorical(np.array(y_train), self.num_classes)

        X_val = self._reshape_data_CNN(X_val)
        y_val = keras.utils.to_categorical(np.array(y_val), self.num_classes)

        tuner = kt.Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=40,
            factor=3
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )

        # Get the optimal hyperparameters
        best_hps= tuner.get_best_hyperparameters(1)[0]

        # get the best model
        best_model = tuner.get_best_models(1)[0]
        self.model = best_model

        self.history = self.model.fit(
            X_train, y_train,
            epochs=100,
            verbose=verbose,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        print("Best hyperparameters:", best_hps.values)

    def predict(self, input: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        '''
        Predict value(s) using the trained CNN
        '''
        if isinstance(input, Image.Image):
            input = self._reshape_image_CNN(input)
            input = np.expand_dims(input, axis=0)
        else:
            input = self._reshape_data_CNN(input)
        return self.model.predict(input)
    
    def evaluate(self, X_test: List[Image.Image], y_test: List[int]) -> dict:
        '''
        Evaluates the model and returns metrics for comparison.
        '''
        CELL_NUM = 81
        # Predict
        y_test = np.array(y_test)
        y_pred_probs = self.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Test accuracy
        test_accuracy = accuracy_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
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
            "sudoku accuracy": correct_percent,
            "precision macro": report["macro avg"]["precision"],
            "recall macro": report["macro avg"]["recall"],
            "f1 macro": report["macro avg"]["f1-score"]
        }, self.history


    def save(self, name: str, path: str=None) -> None:
        if path is None:
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, "saved", "cnn")
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{name}.keras")
        self.model.save(save_path)

    def load(self, name: str, path: str=None) -> None:
        if path is None:
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, "saved", "cnn")
        load_path = os.path.join(path, f"{name}.keras")
        self.model = keras.models.load_model(load_path)
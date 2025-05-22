import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def reshape_image_SVM(img):
    img_array = np.array(img)
    normalized_array = img_array.astype(np.float32) / 255.0
    return normalized_array.flatten()


def reshape_data_SVM(image_list):
    sample = reshape_image_SVM(image_list[0])
    reshaped_data = np.zeros((len(image_list), len(sample)), dtype=np.float32)
    for i, img in enumerate(image_list):
        reshaped_data[i] = reshape_image_SVM(img)
    return reshaped_data


class SVM():
    def __init__(self):
        '''
        Initialise svm as a one versus all SVM
        '''
        # MAYBE ADD THE DIFFERENT SVM PARAMETERS AS ATTRIBUTES?
        self.svm = svm.LinearSVC(verbose=True)

    def train(self, X_train, y_train):
        '''
        Trains svm on X_train matrix and y_train vector
        '''
        X_train = reshape_data_SVM(X_train)
        y_train = np.array(y_train)
        self.svm.fit(X_train, y_train)

    def predict(self, X_test):
        '''
        Predict value(s) using the trained SVM
        '''
        if isinstance(X_test, Image.Image):
            X_test = reshape_image_SVM(X_test)
        else:
            X_test = reshape_data_SVM(X_test)
        return self.svm.predict(X_test)

    def evaluate(self, X_test, y_test):
        '''
        Evaluates the svm using the passed test data
        '''
        y_test = np.array(y_test)
        y_pred = self.predict(X_test)

        # Overall accuracy
        overall_acc = accuracy_score(y_test, y_pred)
        print(f"Overall accuracy: {overall_acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
        disp.plot(cmap='viridis', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()

        # Per class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            print(f"Accuracy for Class {i}: {acc:.4f}")

        # Per class precisionm recall and F1
        report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(10)])
        print(report)

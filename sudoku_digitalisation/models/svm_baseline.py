from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM():
    def __init__(self):
        '''
        Initialise svm as a one versus all SVM
        '''
        self.svm = svm.SVC(decision_function_shape="ovr")

    def train(self, X_train, y_train):
        '''
        Trains svm on X_train matrix and y_train vector
        '''
        self.svm.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        y_prediction = self.svm.predict(X_test)
        print("accuracy: ", accuracy_score(y_test, y_prediction))
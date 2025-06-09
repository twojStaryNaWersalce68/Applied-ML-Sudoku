from sudoku_digitalisation.features.sudoku_preprocessing import get_preprocessor
from sudoku_digitalisation.scripts.run_training import get_model
from sudoku_digitalisation.scripts.run_prediction import make_prediction

# Run: "python -m sudoku_digitalisation.main" if you have ModuleNotFoundError
if __name__ == "__main__":
    IS_PREPROCESSED = True      # bool to see if the dataset is preprocessed already
    TRAIN_CNN = False           # change this to True if you want to train the CNN, otherwise it is loaded
    TRAIN_SVM = False           # change this to True if you want to train the SVM (baseline), otherwise it is loaded
    GET_SVM = False             # if this is set to False, the SVM is not gotten
    EVALUATE = False            # change to true if you want to run the evaluation
    PREDICT = True              # change to true if you want to run the prediction

    CNN_NAME = 'sudoku_cnn'     # name for how you save and load CNN
    SVM_NAME = 'default'        # name for how you save and load SVM

    # preprocessor for everything, clip_limit is for CLAHE, output_size for the output size of cropped images
    preprocessor = get_preprocessor(clip_limit=3, output_size=252, is_preprocessed=IS_PREPROCESSED, path=None)

    # load or train cnn or svm or both
    cnn = get_model(TRAIN_CNN, 'cnn', CNN_NAME, preprocessor)
    if GET_SVM:
        svm = get_model(TRAIN_SVM, 'svm', SVM_NAME, preprocessor)

    # X_test = preprocessor.handler.datasets['digits']['test']['image']
    # y_test = preprocessor.handler.datasets['digits']['test']['label']
    # # evaluate if we want
    # if EVALUATE:
    #     cnn_cm, eval_cnn = cnn.evaluate(X_test, y_test)
    #     cnn_cm.show()
    #     print(eval_cnn)
    #     eval_svm = svm.evaluate(X_test, y_test)
    #     print(eval_svm)
        # if you use an SVM we need to catch the possibility the var has not been assigned.

    # prediction if we want but that would be more for the API
    if PREDICT:
        sudoku_test_set = preprocessor.handler.datasets['raw']['test']
        sample_sudoku = sudoku_test_set[3]
        prediction = make_prediction(sample_sudoku, preprocessor, cnn)
        for i in range(0, 81, 9):
            print(prediction[i:i+9])
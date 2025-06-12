from sudoku_digitalisation.features.sudoku_preprocessing import get_preprocessor
from sudoku_digitalisation.scripts.run_training import get_model
from sudoku_digitalisation.scripts.run_prediction import make_prediction
from sudoku_digitalisation.scripts.run_comparison import compare_cnn_svm
from sudoku_digitalisation.scripts.run_evaluation import evaluate_model

# Run: "python -m sudoku_digitalisation.main" if you have ModuleNotFoundError
if __name__ == "__main__":
    # if you are running it for the first time since my last commit you need to reprocess your data and retrain the CNN!!!!
    IS_PREPROCESSED = True      # bool to see if the dataset is preprocessed already
    TRAIN_CNN = False           # change this to True if you want to train the CNN, otherwise it is loaded
    TRAIN_SVM = False           # change this to True if you want to train the SVM (baseline), otherwise it is loaded
    GET_SVM = True             # if this is set to False, the SVM is not gotten
    EVALUATE = False            # change to true if you want to run the evaluation
    COMPARISON = False          # change to True if you want to compare our CNN model to the SVM baseline
    PREDICT = True              # change to true if you want to run the prediction

    CNN_NAME = '16x2_32x2_128'     # name for how you save and load CNN
    SVM_NAME = 'default'        # name for how you save and load SVM

    # preprocessor for everything, clip_limit is for CLAHE, output_size for the output size of cropped images
    # output size needs to be 450 for the tiny digits to work
    preprocessor = get_preprocessor(clip_limit=3, output_size=450, is_preprocessed=IS_PREPROCESSED, path=None)

    # load or train cnn
    cnn = get_model(TRAIN_CNN, 'cnn', CNN_NAME, preprocessor)

    # perform k-fold CV to get mean accuracy and variance for CNN and SVM
    if COMPARISON:
        compare_cnn_svm(preprocessor, k=5)

    # evaluates selected models individually
    if EVALUATE:
        # TEST AFTER TRAINING CNN
        evaluate_model(cnn, preprocessor, TRAIN_CNN)
        if GET_SVM:
            svm = get_model(TRAIN_SVM, 'svm', SVM_NAME, preprocessor)
            evaluate_model(svm, preprocessor, False)

    # prediction if we want but that would be more for the API
    if PREDICT:
        sudoku_test_set = preprocessor.handler.datasets['raw']['test']
        sample_sudoku = sudoku_test_set['image'][3]
        prediction = make_prediction(sample_sudoku, preprocessor, cnn)
        for i in range(0, 81, 9):
            print(prediction[i:i+9])
        sample_sudoku.show()




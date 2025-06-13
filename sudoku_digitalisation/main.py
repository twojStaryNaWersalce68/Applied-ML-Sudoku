from sudoku_digitalisation.features.sudoku_preprocessing import get_preprocessor
from sudoku_digitalisation.scripts.run_training import get_model
from sudoku_digitalisation.scripts.run_comparison import compare_cnn_svm
from sudoku_digitalisation.scripts.run_evaluation import evaluate_model, evaluate_edge_detection, full_evaluation

# Run: "python -m sudoku_digitalisation.main" if you have ModuleNotFoundError
if __name__ == "__main__":
    IS_PREPROCESSED = False      # bool to see if the dataset is preprocessed already
    TRAIN_CNN = False           # change this to True if you want to train the CNN, otherwise it is loaded
    TRAIN_SVM = False           # change this to True if you want to train the SVM (baseline), otherwise it is loaded
    GET_SVM = True             # if this is set to False, the SVM is not gotten
    EVALUATE = True            # change to true if you want to run the evaluation
    COMPARISON = False          # change to True if you want to compare our CNN model to the SVM baseline

    CNN_NAME = 'tuned_cnn'     # name for how you save and load CNN
    SVM_NAME = 'default'        # name for how you save and load SVM

    # preprocessor for everything, clip_limit is for CLAHE, output_size for the output size of cropped images
    # output size needs to be 450 for the tiny digits to work
    preprocessor = get_preprocessor(clip_limit=3, output_size=450, is_preprocessed=IS_PREPROCESSED, path=None)
    image = preprocessor.handler.datasets['raw']['train']['image'][0]

    # load or train cnn
    cnn = get_model(TRAIN_CNN, 'cnn', CNN_NAME, preprocessor)

    # perform k-fold CV to get mean accuracy and variance for CNN and SVM
    if COMPARISON:
        compare_cnn_svm(preprocessor, k=5)

    # evaluates selected models individually
    if EVALUATE:
        # TEST AFTER TRAINING CNN
        full_evaluation(cnn, preprocessor)
        print("\n CNN EVALUATION:")
        evaluate_model(cnn, preprocessor, TRAIN_CNN)
        print("\n")
        evaluate_edge_detection(preprocessor)
        if GET_SVM:
            print("\n")
            svm = get_model(TRAIN_SVM, 'svm', SVM_NAME, preprocessor)
            evaluate_model(svm, preprocessor, False)

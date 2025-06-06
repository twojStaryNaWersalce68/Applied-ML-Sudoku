from sudoku_digitalisation.features.sudoku_preprocessing import get_preprocessor
from sudoku_digitalisation.run_training import get_model


if __name__ == "__main__":
    IS_PREPROCESSED = True      # bool to see if the dataset is preprocessed already
    CNN = 'cnn'                 # put None if you don't want to get a CNN
    SVM = 'svm'                 # put None if you don't want to get a SVM
    CNN_NAME = 'sudoku_cnn'     # name for saved CNN, None if you want training
    SVM_NAME = None             # name for saved SVM, None if you want training

    # preprocessor for everything, clip_limit is for CLAHE, output_size for the output size of cropped images
    preprocessor = get_preprocessor(clip_limit=3, output_size=252, is_preprocessed=IS_PREPROCESSED)

    # load or train cnn or svm or both
    cnn = get_model(CNN, CNN_NAME, preprocessor)
    svm = get_model(SVM, SVM_NAME, preprocessor)

    # evaluate if we want

    # prediction if we want but that would be more for the API

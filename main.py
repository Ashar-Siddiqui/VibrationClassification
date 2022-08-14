from os.path import dirname, join

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def main():
    # Load preprocessed data
    train0 = np.load(join(dirname(__file__), 'data_tsfresh', '0Dtsfresh.npy'))[:,:,0]
    test0 = np.load(join(dirname(__file__), 'data_tsfresh', '0Etsfresh.npy'))[:,:,0]
    train1 = np.load(join(dirname(__file__), 'data_tsfresh', '4Dtsfresh.npy'))[:,:,0]
    test1 = np.load(join(dirname(__file__), 'data_tsfresh', '4Etsfresh.npy'))[:,:,0]

    # Assemble data into training and testing splits
    train_features = np.concatenate([train0, train1], axis=0)
    train_labels = np.concatenate([np.zeros(train0.shape[0]), np.ones(train1.shape[0])])
    test_features = np.concatenate([test0, test1], axis=0)
    test_labels = np.concatenate([np.zeros(test0.shape[0]), np.ones(test1.shape[0])])
    print(f'Training data rows: {train_features.shape[0]}\n')

    '''
    # These lines of code are not stricly necessary but sometimes helpful

    # Shuffle data so that all of the examples with the same label aren't grouped together
    train_features, train_labels = shuffle(train_features, train_labels, random_state=0)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=0)

    # Normalize data so that each column of values is comparable to the other columns
    # For example, if the columns are on vastly different scales (hundreds vs millions),
    # it is more difficult for some models to learn
    train_features = MinMaxScaler().fit_transform(train_features)
    test_features = MinMaxScaler().fit_transform(test_features)
    '''

    for model in (
        RandomForestClassifier(max_depth=20, min_samples_leaf=1, n_estimators=300, random_state=0, verbose=True),
        GradientBoostingClassifier(random_state=0, n_estimators=10, verbose=True),
        AdaBoostClassifier(random_state=0, n_estimators=10),
        MLPClassifier(random_state=0, hidden_layer_sizes=(20,50,50,20), verbose=True),
        KNeighborsClassifier(n_neighbors=5),
    ):
        print('='*40 + f'\nTraining {model.__class__.__name__}...')
        model.fit(train_features, train_labels)
        model.verbose = False # turn off the print statements that run when the model is used in any way

        predict_train_labels = model.predict(train_features)
        print(f'Accuracy on Training set: {(predict_train_labels == train_labels).mean():.8f}')
        predict_test_labels = model.predict(test_features)
        print(f'Accuracy on Testing set: {(predict_test_labels == test_labels).mean():.8f}')
        if isinstance(model, RandomForestClassifier):
            print(model.feature_importances_)
        print()

if __name__ == '__main__':
    main()

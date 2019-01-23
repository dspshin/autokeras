from keras.datasets import mnist
from autokeras import ImageClassifier
from autokeras.constant import Constant
from autokeras.utils import pickle_from_file
from keras.utils import plot_model

import time

model_file_name = 'mnist.model'


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    print('.')
    start = time.time()

    clf = ImageClassifier(verbose=True, augment=False)

    print('..')
    clf.fit(x_train, y_train, time_limit=30 * 60)

    print('...')
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)

    end = time.time()
    print('Elapsed time:', end-start)

    print('....')
    clf.export_autokeras_model(model_file_name)

    model = pickle_from_file(model_file_name)
    results = model.evaluate(x_test, y_test)
    print(results)

    plot_model(clf, to_file='mnist.png')
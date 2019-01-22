from keras.datasets import mnist
from autokeras import ImageClassifier
from autokeras.constant import Constant
import time

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

    print('....')
    y = clf.evaluate(x_test, y_test)

    print(y * 100)
    end = time.time()
    print('Elased time:', end-start)

    print('.....')
    clf.load_searcher().load_best_model().produce_keras_model().save('my_model.h5')


    print('......')
    from keras.utils import plot_model
    plot_model(clf, to_file='my_model.png')
import load_data
import theano
import numpy as np
import theano.tensor as T
import time
import model_rw

import lasagne
from lasagne.nonlinearities import rectify

def build_simple_cnn(batch_size,input_var=None):

    network = lasagne.layers.InputLayer(shape=(batch_size, 3, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())


    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=6,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


if __name__ == '__main__':
    print("Loading Data")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data.load_dataset("/home/prosurpa/Image/image/")

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Bulding Model")

    batch_size = 1

    network = build_simple_cnn(batch_size, input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9
    )

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


    model_rw.read_model_data(network, "75model")

    for i in range(len(X_train)):
        input_var = [X_train[i]]
        target_var = [y_train[i]]
        print val_fn(input_var, target_var)
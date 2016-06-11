import load_data
import theano
import numpy as np
import theano.tensor as T
import time
import model_rw

import lasagne
from lasagne.nonlinearities import rectify

def build_f_cnn(batch_size, input_var=None):
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
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def build_simple_cnn_(batch_size,input_var=None):

    network = lasagne.layers.InputLayer(shape=(batch_size, 3, 62, 62),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(7, 7),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
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

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=64,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.2),
            num_units=6,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network



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

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx : start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main():

    print("Loading Data")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data.load_data_feautre_train(feautre = u"\uBC18\uD314",root_path= "/home/prosurpa/Image/image/",image_size=(28,28))

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Bulding Model")

    batch_size = 20

    network = build_f_cnn(batch_size ,input_var)


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

    #model_rw.read_model_data(network, "75.0000009934model")

    print("Starting training")


    num_epochs = 1000
    best_acc = 75
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()

        print((len(X_train)/batch_size))
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            if train_batches%20 == 0:
                print(train_batches)



        val_err = 0
        val_acc = 0
        val_batches = 0

        print((len(X_valid) / batch_size))
        for batch in iterate_minibatches(X_valid, y_valid, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            if train_batches % 20 == 0:
                print(val_batches)


        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        test_err = 0
        test_acc = 0
        test_batches = 0

        print((len(X_test) / batch_size))
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
            if train_batches % 20 == 0:
                print(test_batches)


        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

        re_acc = test_acc / test_batches * 100

        if re_acc > best_acc + 0.5:
            best_acc = re_acc
            model_rw.write_model_data(network, str(best_acc) + "model")



if __name__ == '__main__':
    main()
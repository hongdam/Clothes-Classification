import numpy as np
import json
import matplotlib
matplotlib.use("TkAgg")
from PIL import Image

import random
import itertools

def load_data_feautre_train(feautre="" ,root_path = "/Users/HongDam/PycharmProjects/theanoTest/image/",
                image_index_filename = "new_dic_for_search.json",
                image_size=(28, 28),
                dtype='float32'):


    with open(root_path + image_index_filename, 'r') as json_file:
        feautre_dic = json.load(json_file)


    if feautre_dic.get(feautre) == None:
        print("err : No match feautre")
        return

    feautre_list = feautre_dic[unicode(feautre)]
    #feauter_list : list of filename

    #unnecessary
    del feautre_dic[unicode(feautre)]

    another_list = list(itertools.chain.from_iterable(feautre_dic.values()))
    another_list = list(set(another_list) - set(feautre_list))

    random.shuffle(another_list)
    random.shuffle(feautre_list)

    # add symbol
    feautre_list = ["1" + x for x in feautre_list]

    # anoter list : feauter list = 8 : 2
    another_list = another_list[:len(feautre_list)]

    train_index_list = feautre_list[:len(feautre_list)//2] + another_list[:len(another_list)//2]
    valid_index_list = feautre_list[len(feautre_list)//2:-len(feautre_list)//4] + another_list[len(another_list)//2:-len(another_list)//4]
    test_index_list = feautre_list[-len(feautre_list)//4:] + another_list[-len(another_list)//4:]

    random.shuffle(train_index_list)
    random.shuffle(valid_index_list)
    random.shuffle(test_index_list)

    X_train = []
    y_train = []

    X_valid = []
    y_valid = []

    X_test = []
    y_test = []

    def load_image_and_label(filename, dtype='float32'):
        if filename[0] == "1":
            label = 1
            filename = filename[1:]
        else:
            label = 0
        image = Image.open(open(root_path + filename))

        width, height = image.size

        if width != image_size[0] or height != image_size[1]:
            image = image.resize((image_size[0], image_size[1]), Image.ANTIALIAS)

        image = np.asarray(image, dtype=dtype) / 256
        image = image.reshape(1, 3, image_size[0], image_size[1])

        return image.astype(dtype), label


    for index in train_index_list:
        image, label = load_image_and_label(index, dtype)
        X_train.append(image)
        y_train.append(label)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.asarray(y_train)

    for index in valid_index_list:
        image, label = load_image_and_label(index, dtype)
        X_valid.append(image)
        y_valid.append(label)
    X_valid = np.concatenate(X_valid, axis=0)
    y_valid = np.asarray(y_valid)

    for index in test_index_list:
        image, label = load_image_and_label(index, dtype)
        X_test.append(image)
        y_test.append(label)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.asarray(y_test)

    # print X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape

    return X_train, y_train.astype('int32'), X_valid, y_valid.astype('int32'), X_test, y_test.astype('int32')





def load_dataset(root_path = "/Users/HongDam/PycharmProjects/theanoTest/image/",
                 image_index_filename = "new_dic_for_training_list.json",
                 categories_range = [(0, 6997), (6998,12510), (12511,19317), (19318,25029), (25030,28576),(28577,41483)],
                 image_size = (28,28),
                 dtype = 'float32'):

    category_list = ['knit', 'outer', 'pants', 'shirts', 'suit', 'tee']

    with open(root_path + image_index_filename, 'r') as json_file:
        image_filename = json.load(json_file)


    def load_image_and_label(filename, dtype='float32'):

        image = Image.open(open(root_path + filename))

        #resize image
        width, height = image.size
        if width != image_size[0] or height != image_size[1]:
            image = image.resize((image_size[0],image_size[1]), Image.ANTIALIAS)

        image = np.asarray(image, dtype=dtype)/256
        image = image.reshape(1,3,image_size[0],image_size[1])


        #create label
        label = category_list.index(filename.split('/')[0])
        #labal = np.asarray([labal])

        return image.astype(dtype), label


    ## make random index list
    ## train, test, valid
    train_index_list = []
    valid_index_list = []
    test_index_list = []

    for i in range(len(category_list)):
        shuffle_list = range(categories_range[i][0], categories_range[i][1] + 1)
        random.shuffle(shuffle_list)

        valid_index_list.append(shuffle_list[:len(shuffle_list)/12])
        train_index_list.append(shuffle_list[len(valid_index_list[i]):])

    # merge
    valid_index_list = list(itertools.chain.from_iterable(valid_index_list))
    train_index_list = list(itertools.chain.from_iterable(train_index_list))

    # shuffle
    random.shuffle(valid_index_list)
    random.shuffle(train_index_list)

    valid_index_list, test_index_list =  valid_index_list[:-1000], valid_index_list[-1000:]

    #train_index_list, valid_index_list, test_index_list = train_index_list[:50], train_index_list[50:60],train_index_list[60:70]

    X_train = []
    y_train = []

    X_valid = []
    y_valid = []

    X_test = []
    y_test = []

    for index in train_index_list:
        image, label = load_image_and_label(image_filename[index], dtype)
        X_train.append(image)
        y_train.append(label)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.asarray(y_train)

    for index in valid_index_list:
        image, label = load_image_and_label(image_filename[index], dtype)
        X_valid.append(image)
        y_valid.append(label)
    X_valid = np.concatenate(X_valid, axis=0)
    y_valid = np.asarray(y_valid)

    for index in test_index_list:
        image, label = load_image_and_label(image_filename[index], dtype)
        X_test.append(image)
        y_test.append(label)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.asarray(y_test)

    #print X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape

    return X_train, y_train.astype('int32'), X_valid, y_valid.astype('int32'), X_test, y_test.astype('int32')


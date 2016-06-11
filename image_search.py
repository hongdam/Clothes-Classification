import theano
import theano.tensor as T
import lasagne

import matplotlib
matplotlib.use("TkAgg")
from PIL import Image

import numpy as np

import model_rw

import json

def build_simple_cnn(input_shape,num_of_output ,input_var = None):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)


    network = lasagne.layers.Conv2DLayer(network, 32, (5,5))
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))

    network = lasagne.layers.Conv2DLayer(network, 32, (5,5))
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network,p=0.5),
                                        num_units=256)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=0.5),
                                        num_units=num_of_output, nonlinearity=lasagne.nonlinearities.softmax)

    return network

def main():
    pass

if __name__ == '__main__':

    category_list = ['knit', 'outer', 'pants', 'shirts', 'suit', 'tee']
    print("Please choose category 1.knit 2.outer 3.pants 4.shirts 5.suit 6.tee")
    path = int(raw_input())
    #path = "/Users/HongDam/Downloads/ClothingAttributeDataset/images/"
    path = "/Users/HongDam/PycharmProjects/theanoTest/image/" + category_list[path-1]

    print("Please input filename")
    filename = raw_input()
    #filename = "001628.jpg"
    filename = "/" + filename

    print(path+filename)

    input_image = Image.open(open(path+filename))
    input_image.show()

    width, height = input_image.size

    if width != 28 and height != 28:
        input_image = input_image.resize((28,28), Image.ANTIALIAS)

    input_image = np.asarray(input_image, dtype='float32') / 256
    input_image = input_image.reshape(1, 3, 28,28)

    input_var = T.tensor4()

    input_shape = (1,3,28,28)
    # create CNN
    category_network = build_simple_cnn(input_shape,6, input_var)
    ban_network = build_simple_cnn(input_shape,2, input_var)
    check_network = build_simple_cnn(input_shape,2,input_var)
    denim_network = build_simple_cnn(input_shape, 2, input_var)
    padding_network = build_simple_cnn(input_shape, 2, input_var)
    flower_network = build_simple_cnn(input_shape, 2, input_var)
    blue_network = build_simple_cnn(input_shape, 2, input_var)


    # load CNN
    modelfile_path = "./category_83.1"
    model_rw.read_model_data(category_network, modelfile_path)
    modelfile_path = "./ban_95.8model"
    model_rw.read_model_data(ban_network, modelfile_path)
    modelfile_path = "./check_83.9"
    model_rw.read_model_data(check_network, modelfile_path)
    modelfile_path = "./denim_84.0"
    model_rw.read_model_data(denim_network, modelfile_path)
    modelfile_path = "./padding_93.1"
    model_rw.read_model_data(padding_network, modelfile_path)
    modelfile_path = "./flower_75.0model"
    model_rw.read_model_data(flower_network, modelfile_path)
    modelfile_path = "./blue_94.1666672627model"
    model_rw.read_model_data(blue_network, modelfile_path)


    # output layer
    category_pre = lasagne.layers.get_output(category_network,input_var)
    ban_pre = lasagne.layers.get_output(ban_network, input_var)
    check_pre = lasagne.layers.get_output(check_network, input_var)
    denim_pre = lasagne.layers.get_output(denim_network, input_var)
    padding_pre = lasagne.layers.get_output(padding_network, input_var)
    flower_pre = lasagne.layers.get_output(flower_network, input_var)
    blue_pre = lasagne.layers.get_output(blue_network, input_var)


    # create theano function
    cate_ftn = theano.function([input_var], category_pre)
    cate = cate_ftn(input_image)

    ban_ftn = theano.function([input_var], ban_pre)
    ban = ban_ftn(input_image)

    check_ftn = theano.function([input_var], check_pre)
    check = check_ftn(input_image)

    denim_ftn = theano.function([input_var], denim_pre)
    denim = denim_ftn(input_image)

    padding_ftn = theano.function([input_var], padding_pre)
    padding = padding_ftn(input_image)

    flower_ftn = theano.function([input_var], flower_pre)
    flower = flower_ftn(input_image)

    blue_ftn = theano.function([input_var], blue_pre)
    blue = blue_ftn(input_image)

    #

    o_category = category_list[cate.argmax()]

    f = lambda x: True if x.argmax() == 1 else False
    feature_list = [u"\uBC18\uD314", u"\uCCB4\uD06C", u"\ub370\ub2d8", u"\uD328\uB529", u"\ud50c\ub77c\uc6cc", u"\uBE14\uB8E8\uC885"]
    extracted_features_list = []
    extracted_features_list.append(f(ban))
    extracted_features_list.append(f(check))
    extracted_features_list.append(f(denim))
    extracted_features_list.append(f(padding))
    extracted_features_list.append(f(flower))
    extracted_features_list.append(f(blue))

    #

    #category_list = ['knit', 'outer', 'pants', 'shirts', 'suit', 'tee']
    print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(*cate[0]))
    print("half_sleeve :\t{:.2f} {:.2f}".format(*ban[0]))
    print("check :\t\t\t{:.2f} {:.2f}".format(*check[0]))
    print("denim :\t\t\t{:.2f} {:.2f}".format(*denim[0]))
    print("padding :\t\t\t{:.2f} {:.2f}".format(*padding[0]))
    print("flower :\t\t\t{:.2f} {:.2f}".format(*flower[0]))
    print("blue :\t\t\t{:.2f} {:.2f}".format(*blue[0]))
    #print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))


    with open("/Users/HongDam/PycharmProjects/theanoTest/image/new_dic_for_search.json", 'r') as json_file:
        search_dic = json.load(json_file)

    result_list = []

    print extracted_features_list

    # intersection
    for i in range(len(extracted_features_list)):
        if extracted_features_list[i] == True:
            if len(result_list) != 0:
                result_list = list(set(result_list) & set(search_dic[feature_list[i]]))
            else:
                result_list = search_dic[feature_list[i]]

    # category check
    R = []
    for i in range(len(result_list)):
        if result_list[i].split("/")[0] == o_category:
            R.append(result_list[i])

    print "num of reslut : ",len(R)

    path = "/Users/HongDam/PycharmProjects/theanoTest/image/"

    np.set_printoptions(threshold='nan')

    #print(category_network.W.get_value().shape)
    print(lasagne.layers.get_all_param_values(category_network))

    # show maximum 5 image

    for i in range(len(R[:10])):
        print R[i]
        im = Image.open(open(path + R[i]))
        #im.show()





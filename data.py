#Importamos las librerias necesariasimport pickle
import numpy as np
import os
import sys

"""
    En este proyecto los datos se tienen almacenados en 6 archivos, en este archivo
    se leen esos archivos y se regresan los datos de forma que se puedan utilizar
"""
def get_data_set(name):
    x = None
    y = None
    l = None
    
    #Se obtiene el nombre de cada clase
    f = open('./data_set/cifar_10/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    #Se obtienen los datos del train
    if name is "train":
        for i in range(5):
            f = open('./data_set/cifar_10/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    #Se obtienen los datos del test
    elif name is "test":
        f = open('./data_set/cifar_10/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    #Los outputs se pasan a vectores binarios
    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    return x, dense_to_one_hot(y), l

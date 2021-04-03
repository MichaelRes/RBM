from principal_RBM_alpha import init_RBM, entree_sortie_RBM, \
                                train_RBM, sortie_entree_RBM
import numpy as np
import random
import matplotlib.pyplot as plt


class DNN():
    def __init__(self, rbms):
        self.rbms = rbms


def init_DNN(size_v):
    rbms = []
    for i in range(len(size_v)-1):
        rbms.append(init_RBM(size_v[i], size_v[i+1]))  # adapting dimensions of the rbms

    dbn = DNN(rbms)
    return dbn


def pretrain_DNN(dbn, n_epochs=10, lr=0.1, batch_size=64, X=None, print_error=True):
    for i in range(len(dbn.rbms)):
        dbn.rbms[i] = train_RBM(dbn.rbms[i], n_epochs, lr, batch_size, X, print_error)
        X = entree_sortie_RBM(dbn.rbms[i], X)

    return dbn


def generer_image_DBN(dnn, n_iter, n_images):
    images = []
    image_size = len(dnn.rbms[0].a)
    for k in range(n_images):
        image = np.random.randint(0, 2, image_size)
        for i in range(n_iter):
            for j in range(len(dnn.rbms)):
                proba_sortie = entree_sortie_RBM(dnn.rbms[j], image)
                image = np.random.binomial(1, proba_sortie)

            for j in range(len(dnn.rbms)):
                proba_entree = sortie_entree_RBM(dnn.rbms[-j-1], image)
                image = np.random.binomial(1, proba_entree)

        images.append(image)

    images = np.array(images).reshape((-1, 20, 16))
    for image in images:
        plt.imshow(image)
        plt.show()
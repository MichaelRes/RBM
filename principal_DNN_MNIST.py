from principal_RBM_alpha import entree_sortie_RBM
import numpy as np


def calcul_softmax(rbm, X):
    sortie = entree_sortie_RBM(rbm, X)
    res = np.exp(sortie)/sum(np.exp(sortie))
    return res


def entree_sortie_reseau(dnn, X):
    res = []
    for i in range(len(dnn.rbms)-1):
        X = entree_sortie_RBM(dnn.rbms[i], X)
        res.append(X)
    res.append(calcul_softmax(dnn.rbms[-1]), X)
    return res


def retropropagation(dnn, n_epochs=10, lr=0.1, batch_size=64, X=None, y=None):
    return


def test_DNN():
    return
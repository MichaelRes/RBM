from principal_RBM_alpha import entree_sortie_RBM
import numpy as np


def calcul_softmax(rbm, X):
    sortie = X @ rbm.W + rbm.b
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

    N = len(X)
    n_batch = int(N//batch_size)
    for _ in range(n_epochs):
        np.random.shuffle(X)
        for k in range(n_batch):
            layers = []
            x_0 = X[k*batch_size: (k+1)*batch_size]
            layers.append(x_0)

            #FORWARD
            for (i, rbm) in enumerate(dnn.rbms):
                if i == len(dnn.rbms)-1:
                    sortie = calcul_softmax(rbm, layers[-1])
                else:
                    sortie = entree_sortie_RBM(rbm, layers[-1])

            layers.append(sortie)

            #BACKWARD
            c = layers[-1] - y
            for i in range(1, len(dnn.rbms) + 1):
                db = np.sum(c, axis = 0):
                dW = c.T @ layers[-1-i]
                c = (c @ dnn.rbms[-i].W) * layers[-1-i] * (1 - layers[-1-i])
                dnn.rbms[i].W -= lr * dW
                dnn.rbms[i].b -= lr * db
    
    return dnn

def test_DNN(dnn, X_valid, y_valid):
    y_pred = np.argmax(entree_sortie_reseau(dnn, X_valid)[-1], axis=0)
    erreur = np.mean(y_pred == y_valid)

    return erreur

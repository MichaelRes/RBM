from principal_RBM_alpha import entree_sortie_RBM
import numpy as np
import matplotlib.pyplot as plt


def calcul_softmax(rbm, X):
    sortie = X @ rbm.W + rbm.b
    sortie = sortie - sortie.max()
    res = np.exp(sortie)
    return res/np.sum(res)


def entree_sortie_reseau(dnn, X):
    layers = [X]
    for (i, rbm) in enumerate(dnn.rbms):
        if i == len(dnn.rbms)-1:
            sortie = calcul_softmax(rbm, layers[-1])
        else:
            sortie = entree_sortie_RBM(rbm, layers[-1])
        layers.append(sortie)
    return layers


def retropropagation(dnn, n_epochs=10, lr=0.1, batch_size=64, X_train=None, y_train=None, X_valid=None, y_valid=None, plot_error=False):
    N = len(X_train)
    n_batch = int(N//batch_size)
    errors = []
    for l in range(n_epochs):
        print("Epoch %s / %s" % (l+1, n_epochs))
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        for k in range(n_batch):
            x_0_batch = X_train[indexes[k*batch_size: (k+1)*batch_size]]
            y_batch = y_train[indexes[k*batch_size: (k+1)*batch_size]]
            y_onehot_batch = np.zeros((batch_size, 10))
            y_onehot_batch[np.arange(batch_size), y_batch] = 1

            #FORWARD
            layers = entree_sortie_reseau(dnn, x_0_batch)

            #BACKWARD
            preds_index = np.argmax(layers[-1], axis =1)
            preds_onehot = np.zeros((batch_size, 10))
            preds_onehot[np.arange(batch_size), preds_index] = 1
            c = preds_onehot - y_onehot_batch
            for i in range(1, len(dnn.rbms) + 1):
                db = np.mean(c, axis = 0)
                dW = 1/dnn.rbms[-i].W.shape[0] * layers[-1-i].T @ c
                c = (c @ dnn.rbms[-i].W.T) * layers[-1-i] * (1 - layers[-1-i])
                dnn.rbms[-i].W -= lr * dW
                dnn.rbms[-i].b -= lr * db
        erreur = test_DNN(dnn, X_valid, y_valid)
        print('erreur: ', erreur)
        errors.append(erreur)
    if plot_error:
        plt.plot(np.arange(1, n_epochs+1), errors)
        plt.title('Validation Error rate vs. number of Epochs', fontsize=16)
        plt.xlabel('number of Epochs', fontsize=12)
        plt.ylabel('Validation Error rate', fontsize=12)
        plt.savefig('Fig4.pdf')
        plt.show()

    return dnn

def test_DNN(dnn, X_valid, y_valid):
    results = entree_sortie_reseau(dnn, X_valid)[-1]
    y_pred = np.argmax(results, axis=1)
    erreur = np.mean(y_pred != y_valid)

    return erreur
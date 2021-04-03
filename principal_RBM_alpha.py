import numpy as np
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt

image_size = (20, 16)  # for alphadigit
data = loadmat('binaryalphadigs.mat')


class RBM():
    def __init__(self, W, a, b):
        self.W = W
        self.a = a
        self.b = b


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def lire_alpha_digit(L):
    n = len(L)
    p = data['dat'].shape[1]
    dim1, dim2 = data['dat'][0][0].shape
    X = np.zeros((n*p, dim1*dim2))
    i = 0
    for _, im in np.ndenumerate(data['dat'][L]):
        X[i, :] = im.flatten()
        i += 1
    return X


def init_RBM(n_visible, n_hidden):
    W = np.random.normal(0, 0.1, (n_visible, n_hidden))
    a = np.zeros((n_visible))
    b = np.zeros((n_hidden))
    rbm = RBM(W, a, b)
    return rbm


def entree_sortie_RBM(rbm, X):
    return sigmoid(X @ rbm.W + rbm.b)


def sortie_entree_RBM(rbm, hid):
    return sigmoid(hid @ rbm.W.T + rbm.a)


def train_RBM(rbm, n_epochs=10, lr=0.1, batch_size=64, X=None):

    N = len(X)
    n_batch = int(N//batch_size)
    for _ in range(n_epochs):
        np.random.shuffle(X)
        for k in range(n_batch):
            x_0 = X[k*batch_size: (k+1)*batch_size]
            p_h_v_0 = entree_sortie_RBM(rbm, x_0)
            h_0 = np.random.binomial(1, p_h_v_0)
            pos_e = x_0.T @ p_h_v_0
            p_v_h_0 = sortie_entree_RBM(rbm, h_0)
            x_1 = np.random.binomial(1, p_v_h_0)
            p_h_v_1 = entree_sortie_RBM(rbm, x_1)
            neg_e = x_1.T @ p_h_v_1
            rbm.W += lr*(pos_e - neg_e)
            rbm.a += lr*(x_0 - x_1).sum(axis=0)
            rbm.b += lr*(p_h_v_0 - p_h_v_1).sum(axis=0)
            error = np.sum((x_1 - x_0)**2)
            print('Squarred Error: ', error)
    return rbm


def generer_image_RBM(rbm, n_iter, n_images):
    images = []
    image_size = len(rbm.a)
    for k in range(n_images):
        image = np.random.randint(0, 2, image_size)
        for i in range(n_iter):
            p_h = entree_sortie_RBM(rbm, image)
            h = np.array(np.random.binomial(1, p_h))
            p_v = sortie_entree_RBM(rbm, h)
            v = np.array(np.random.binomial(1, p_v))

        images.append(v)
    images = np.array(images).reshape((-1, 20, 16))
    for image in images:
        plt.imshow(image)
        plt.show()

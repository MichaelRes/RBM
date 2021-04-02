from principal_RBM_alpha import init_RBM, entree_sortie_RBM, train_RBM

class DBN():
    def __init__(self, rbms):
        self.rbms = rbms


def init_DNN(n_layers, size_v):
    rbms = []
    for i in range(n_layers-1):
        rbms.append(init_RBM(size_v[i], size_v[i+1]))  # adapting dimensions of the rbms

    dbn = DBN(rbms)
    return dbn


def pretrain_DNN(dbn, n_epochs=10, lr=0.1, batch_size=64, X=None):
    for i in range(len(dbn.rbms)):
        train_RBM(dbn.rbm[i], n_epochs, lr, batch_size, X)
        X = entree_sortie_RBM(dbn.rbm[i], X)

    return dbn


def generer_image_DBN(dnn, n_iter, n_images):
    images = []
    image_size = len(dnn.rbm[-1].a
    for k in range(n_images):
        image = np.random.randint(random.randint(0,2,image_size)
        for i in range(n_iter):
            proba_sortie = entree_sortie_RBM(dnn.rbm[0], image)
            sortie = np.array([np.random.binomial(1, p) for p in proba_sortie])
            pos_e = sortie @ sortie.T
            proba_entree = sortie_entree_RBM(dnn.rbm[0], sortie)
            image = np.array([np.random.binomial(1, p) for p in proba_entree])

        for j in range(len(dbn.rbms)):
            image = entree_sortie_RBM(dbn.rbm[i], image)

        images.append(image)
    images = np.array(images).reshape((-1, 20, 16))
    for image in images:
        plt.imshow(image)
        plt.show()
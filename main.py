from principal_RBM_alpha import lire_alpha_digit, init_RBM, \
                                train_RBM, generer_image_RBM

from principal_DBN_alpha import init_DNN, \
                                pretrain_DNN, generer_image_DBN

from principal_DNN_MNIST import retropropagation, test_DNN

from sklearn.model_selection import train_test_split
from mlxtend.data import loadlocal_mnist



config_rbm_alpha = {
    'size_v': [320, 100, 100],
    'n_iter': 50,
    'lr': 0.01,
    'batch_size': 30,
    'n_epochs': 1000,
    'L': [8],
    'n_images': 4,
    'show': False
}

config_dbn_alpha = {
    'size_v': [320, 100, 100],
    'n_iter': 50,
    'lr': 0.01,
    'batch_size': 30,
    'n_epochs': 2000,
    'L': [6],
    'n_images': 4,
    'show': False
}


config_dbn_mnist = {
    'size_v': [784, 100, 100],
    'n_iter': 50,
    'lr': 0.01,
    'batch_size': 30,
    'n_epochs': 10,
    'L': [6],
    'n_images': 4,
    'show': False
}

config_dnn_mnist = {
    'size_v': [784, 256, 10],
    'n_iter': 50,
    'lr': 0.01,
    'batch_size': 30,
    'n_epochs': 10,
    'L': [6],
    'n_images': 4,
    'show': True
}


if __name__ == "__main__":
    if config_rbm_alpha['show']:
        print('-------------Entrainement rbm-------------')
        X = lire_alpha_digit(config_rbm['L'])
        rbm = init_RBM(config_rbm['size_v'][0], config_rbm['size_v'][1])
        rbm = train_RBM(rbm,
                        n_epochs=config_rbm['n_epochs'],
                        lr=config_rbm['lr'],
                        batch_size=config_rbm['batch_size'],
                        X=X)

        generer_image_RBM(rbm, config_rbm['n_iter'], 1)

    if config_dbn_alpha['show']:
        print('-------------Entrainement dbn-------------')
        X = lire_alpha_digit(config_dbn['L'])
        dbn = init_DNN(config_dbn['size_v'])
        dbn = pretrain_DNN(dbn,
                           n_epochs=config_dbn['n_epochs'],
                           lr=config_dbn['lr'],
                           batch_size=config_dbn['batch_size'],
                           X=X)

        generer_image_DBN(dbn, config_dbn['n_iter'], 4)

    # Main analysis
    if config_dbn_mnist['show']:
        X, y = loadlocal_mnist(images_path='train-images-idx3-ubyte',
                            labels_path='train-labels-idx1-ubyte')
        X = (X > 0).astype(int)  # convert to binary images

        # initialize dnns
        pretrained_dnn = init_DNN(config_mnist['size_v'])
        pretrained_dnn = pretrain_DNN(pretrained_dnn,
                        n_epochs=config_dbn['n_epochs'],
                        lr=config_dbn['lr'],
                        batch_size=config_dbn['batch_size'],
                        X=X)

        dnn = init_DNN(config_mnist['size_v'])


    if config_dnn_mnist['show']:
        print('-------------Retropropagation DNN-------------')
        X, y = loadlocal_mnist(images_path='train-images-idx3-ubyte',
                            labels_path='train-labels-idx1-ubyte')
        X = (X > 0).astype(int)  # convert to binary images
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

        dnn = init_DNN(config_dnn_mnist['size_v'])
        trained_dnn = retropropagation(dnn, n_epochs=config_dnn_mnist['n_epochs'], lr=config_dnn_mnist['lr'], batch_size=config_dnn_mnist['batch_size'], X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
        erreur = test_DNN(trained_dnn, X_valid, y_valid)
        print('erreur: ', erreur)

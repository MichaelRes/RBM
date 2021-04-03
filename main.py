from principal_RBM_alpha import lire_alpha_digit, init_RBM, \
                                train_RBM, generer_image_RBM

from principal_DBN_alpha import init_DNN, \
                                pretrain_DNN, generer_image_DBN

from principal_DNN_MNIST import retropropagation, test_DNN

from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

config_rbm = {
    'size_v': [320, 100, 100],
    'n_iter': 50,
    'lr': 0.01,
    'batch_size': 30,
    'n_epochs': 1000,
    'n_data': 5,
    'L': [8],
    'n_images': 4,
    'show': False
}

config_dbn = {
    'size_v': [320, 100, 100],
    'n_iter': 50,
    'lr': 0.01,
    'batch_size': 30,
    'n_epochs': 2000,
    'n_data': 5,
    'L': [6],
    'n_images': 4,
    'show': False
}

config = {
    'size_v': [784, 100, 10],
    'n_iter': 50,
    'lr': 0.01,
    'batch_size': 1000,
    'n_epochs': 1,
    'n_data': 5,
    'L': [6],
    'n_images': 4,
    'show': False
}




if __name__ == "__main__":
    if config_rbm['show']:
        print('-------------Entrainement rbm-------------')
        X = lire_alpha_digit(config_rbm['L'])
        rbm = init_RBM(config_rbm['size_v'][0], config_rbm['size_v'][1])
        rbm = train_RBM(rbm,
                        n_epochs=config_rbm['n_epochs'],
                        lr=config_rbm['lr'],
                        batch_size=config_rbm['batch_size'],
                        X=X)

        generer_image_RBM(rbm, config_rbm['n_iter'], 1)

    if config_dbn['show']:
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
    X, y = loadlocal_mnist(images_path='train-images-idx3-ubyte',
                           labels_path='train-labels-idx1-ubyte')
    X = (X > 0).astype(int)  # convert to binary images
    enc = OneHotEncoder(handle_unknown='ignore')
    y = enc.fit_transform(y.reshape(-1,1)).toarray()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


    # initialize dnns
    pretrained_dnn = init_DNN(config['size_v'])
    pretrained_dnn = pretrain_DNN(pretrained_dnn,
                       n_epochs=config['n_epochs'],
                       lr=config['lr'],
                       batch_size=config['batch_size'],
                       X=X_train)

    dnn = init_DNN(config['size_v'])
    dnn = retropropagation(dnn, n_epochs = config['n_epochs'],
                           lr = config['lr'], batch_size = config['batch_size'],
                           X=X_train, y=y_train)

    pretrained_dnn = retropropagation(pretrained_dnn, n_epochs = config['n_epochs'],
                           lr = config['lr'], batch_size = config['batch_size'],
                           X=X_train, y=y_train)

    erreur = test_DNN(pretrained_dnn, X_valid, y_valid)
    erreur_pretrained = test_DNN(pretrained_dnn, X_valid, y_valid)
    print('erreur: ', erreur)
    print('erreur_pretrained: ', erreur_pretrained)





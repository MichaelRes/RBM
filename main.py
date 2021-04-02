from principal_RBM_alpha import lire_alpha_digit, init_RBM, \
                                train_RBM, generer_image_RBM

from principal_DBN_alpha import init_DNN, \
                                pretrain_DNN, generer_image_DBN

size_v = [320, 100, 100]
n_iter = 50
lr = 0.01
batch_size = 30
n_epochs = 1000
n_data = 5
L = [8]  # characters to learn


if __name__ == "__main__":
    X = lire_alpha_digit(L)
    # rbm = init_RBM(size_v[0], size_v[1])
    # rbm = train_RBM(rbm, n_epochs=n_epochs, lr=lr, batch_size=batch_size, X=X)
    # generer_image_RBM(rbm, n_iter, 1)

    dbn = init_DNN(size_v)
    rbm = pretrain_DNN(dbn, n_epochs=n_epochs, lr=lr, batch_size=batch_size, X=X)
    generer_image_DBN(dbn, n_iter, 4)

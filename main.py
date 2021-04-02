from principal_RBM_alpha import lire_alpha_digit, init_RBM, \
                                train_RBM, generer_image_RBM

size_v = [320, 100, 320, 200, 320, 200, 320]
n_iter = 100
lr = 0.1
batch_size = 20
n_epochs = 50
n_data = 5
L = [0, 10]  # characters to learn


if __name__ == "__main__":
    X = lire_alpha_digit(L)
    rbm = init_RBM(size_v[0], size_v[1])
    rbm = train_RBM(rbm, n_epochs=n_epochs, lr=lr, batch_size=batch_size, X=X)
    generer_image_RBM(rbm, n_iter, 5)

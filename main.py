from principal_RBM_alpha import lire_alpha_digit, init_RBM, \
                                train_RBM, generer_image_RBM

from principal_DBN_alpha import init_DNN, \
                                pretrain_DNN, generer_image_DBN

from principal_DNN_MNIST import retropropagation, test_DNN

from sklearn.model_selection import train_test_split
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt



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
    'size_v': [784, 256, 128],
    'lr': 0.01,
    'batch_size': 100,
    'n_epochs': 1,
}

config_dnn_mnist = {
    'size_v': [784, 256, 128, 10],
    'lr': 0.01,
    'batch_size': 100,
    'n_epochs': 10,
}

input_dim = 784
n_classes_mnist = 10

generate_plots = [False, False, False] # Set to true to generate the corresponding figure
show_best_model = True  # Set to true to create and train the model with the lowest error (and plot the evolution of the error)

if __name__ == "__main__":
    if config_rbm_alpha['show']:
        print('-------------Training rbm-------------')
        X = lire_alpha_digit(config_rbm_alpha['L'])
        rbm = init_RBM(config_rbm_alpha['size_v'][0], config_rbm_alpha['size_v'][1])
        rbm = train_RBM(rbm,
                        n_epochs=config_rbm_alpha['n_epochs'],
                        lr=config_rbm_alpha['lr'],
                        batch_size=config_rbm_alpha['batch_size'],
                        X=X)

        generer_image_RBM(rbm, config_rbm_alpha['n_iter'], 1)

    if config_dbn_alpha['show']:
        print('-------------Training dbn-------------')
        X = lire_alpha_digit(config_dbn_alpha['L'])
        dbn = init_DNN(config_dbn_alpha['size_v'])
        dbn = pretrain_DNN(dbn,
                           n_epochs=config_dbn_alpha['n_epochs'],
                           lr=config_dbn_alpha['lr'],
                           batch_size=config_dbn_alpha['batch_size'],
                           X=X)

        generer_image_DBN(dbn, config_dbn_alpha['n_iter'], 4)

    # Main analysis

    # Loading dataset
    X, y = loadlocal_mnist(images_path='train-images-idx3-ubyte',
                        labels_path='train-labels-idx1-ubyte')
    X = (X > 0).astype(int)  # convert to binary images
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    # First plot: Error as a function of the number of layers

    if generate_plots[0]:
        errors_pretrained_dnn = []
        errors_dnn = []
        n_layers_v = [2,3,4,5]
        
        for n_layers in n_layers_v:
            print('Number of layers: ', n_layers)
            size_v = [input_dim] + [200 for i in range(n_layers)]

            # Initialize DNNs
            pretrained_dnn = init_DNN(size_v)
            dnn = init_DNN(size_v + [n_classes_mnist])

            print('-------------Pretraining DNN-------------')
            pretrained_dnn = pretrain_DNN(pretrained_dnn,
                            n_epochs=config_dbn_mnist['n_epochs'],
                            lr=config_dbn_mnist['lr'],
                            batch_size=config_dbn_mnist['batch_size'],
                            X=X, print_error=False)
            pretrained_dnn.rbms.append(init_RBM(200, n_classes_mnist)) # Add classification layer

            print('-------------Retropropagation Pretrained DNN-------------')
            pretrained_dnn = retropropagation(pretrained_dnn, n_epochs=config_dnn_mnist['n_epochs'], lr=config_dnn_mnist['lr'], batch_size=config_dnn_mnist['batch_size'], X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
            error_pretrained_dnn = test_DNN(pretrained_dnn, X_valid, y_valid)
            errors_pretrained_dnn.append(error_pretrained_dnn)
            print('Error Pretrained DNN: ', error_pretrained_dnn)


            print('-------------Retropropagation DNN-------------')
            dnn = retropropagation(dnn, n_epochs=config_dnn_mnist['n_epochs'], lr=config_dnn_mnist['lr'], batch_size=config_dnn_mnist['batch_size'], X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
            error_dnn = test_DNN(dnn, X_valid, y_valid)
            errors_dnn.append(error_dnn)
            print('Error DNN: ', error_dnn)
        
        plt.plot(n_layers_v, errors_pretrained_dnn)
        plt.plot(n_layers_v, errors_dnn)
        plt.legend(["pretrained DNN", "DNN"])
        plt.title('Error rate vs. number of Layers', fontsize=16)
        plt.xlabel('Number of layers', fontsize=12)
        plt.ylabel('Error rate', fontsize=12)
        plt.savefig('Fig1.pdf')
        plt.show()
    
    # Second plot: Error as a function of the number of neurons

    if generate_plots[1]:
        errors_pretrained_dnn = []
        errors_dnn = []
        n_neurons_v = [100,300,500,700]
        
        for n_neurons in n_neurons_v:
            print('Number of neurons: ', n_neurons)
            size_v = [input_dim] + [n_neurons, n_neurons]

            # Initialize DNNs
            pretrained_dnn = init_DNN(size_v)
            dnn = init_DNN(size_v + [n_classes_mnist])

            print('-------------Pretraining DNN-------------')
            pretrained_dnn = pretrain_DNN(pretrained_dnn,
                            n_epochs=config_dbn_mnist['n_epochs'],
                            lr=config_dbn_mnist['lr'],
                            batch_size=config_dbn_mnist['batch_size'],
                            X=X, print_error=False)
            pretrained_dnn.rbms.append(init_RBM(n_neurons, n_classes_mnist)) # Add classification layer

            print('-------------Retropropagation Pretrained DNN-------------')
            pretrained_dnn = retropropagation(pretrained_dnn, n_epochs=config_dnn_mnist['n_epochs'], lr=config_dnn_mnist['lr'], batch_size=config_dnn_mnist['batch_size'], X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
            error_pretrained_dnn = test_DNN(pretrained_dnn, X_valid, y_valid)
            errors_pretrained_dnn.append(error_pretrained_dnn)
            print('Error Pretrained DNN: ', error_pretrained_dnn)


            print('-------------Retropropagation DNN-------------')
            dnn = retropropagation(dnn, n_epochs=config_dnn_mnist['n_epochs'], lr=config_dnn_mnist['lr'], batch_size=config_dnn_mnist['batch_size'], X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
            error_dnn = test_DNN(dnn, X_valid, y_valid)
            errors_dnn.append(error_dnn)
            print('Error DNN: ', error_dnn)
        
        plt.plot(n_neurons_v, errors_pretrained_dnn)
        plt.plot(n_neurons_v, errors_dnn)
        plt.legend(["pretrained DNN", "DNN"])
        plt.title('Error rate vs. number of Neurons', fontsize=16)
        plt.xlabel('Number of Neurons', fontsize=12)
        plt.ylabel('Error rate', fontsize=12)
        plt.savefig('Fig2.pdf')
        plt.show()


    # Third plot: Error as a function of the number of data

    if generate_plots[2]:
        errors_pretrained_dnn = []
        errors_dnn = []
        n_data_v = [1000,3000,7000,10000,30000,48000]
        size_v = [input_dim] + [200,200]

        for n_data in n_data_v:
            print('Number of data: ', n_data)
            # Initialize DNNs
            pretrained_dnn = init_DNN(size_v)
            dnn = init_DNN(size_v + [n_classes_mnist])

            print('-------------Pretraining DNN-------------')
            pretrained_dnn = pretrain_DNN(pretrained_dnn,
                            n_epochs=config_dbn_mnist['n_epochs'],
                            lr=config_dbn_mnist['lr'],
                            batch_size=config_dbn_mnist['batch_size'],
                            X=X, print_error=False)
            pretrained_dnn.rbms.append(init_RBM(200, n_classes_mnist)) # Add classification layer

            print('-------------Retropropagation Pretrained DNN-------------')
            pretrained_dnn = retropropagation(pretrained_dnn, n_epochs=config_dnn_mnist['n_epochs'], lr=config_dnn_mnist['lr'], batch_size=config_dnn_mnist['batch_size'], X_train=X_train[:n_data], y_train=y_train[:n_data], X_valid=X_valid, y_valid=y_valid)
            error_pretrained_dnn = test_DNN(pretrained_dnn, X_valid, y_valid)
            errors_pretrained_dnn.append(error_pretrained_dnn)
            print('Error Pretrained DNN: ', error_pretrained_dnn)


            print('-------------Retropropagation DNN-------------')
            dnn = retropropagation(dnn, n_epochs=config_dnn_mnist['n_epochs'], lr=config_dnn_mnist['lr'], batch_size=config_dnn_mnist['batch_size'], X_train=X_train[:n_data], y_train=y_train[:n_data], X_valid=X_valid, y_valid=y_valid)
            error_dnn = test_DNN(dnn, X_valid, y_valid)
            errors_dnn.append(error_dnn)
            print('Error DNN: ', error_dnn)
        
        plt.plot(n_data_v, errors_pretrained_dnn)
        plt.plot(n_data_v, errors_dnn)
        plt.legend(["pretrained DNN", "DNN"])
        plt.title('Error rate vs. number of Data', fontsize=16)
        plt.xlabel('Number of Data', fontsize=12)
        plt.ylabel('Error rate', fontsize=12)
        plt.savefig('Fig3.pdf')
        plt.show()

    # Looking for the best configuration:
    if show_best_model:
        best_config = {
        'n_epochs_pretrain': 10,
        'batch_size_pretrain': 25,
        'lr_pretrain': 0.01,
        'size_v': [784, 200, 200, 200, 200],
        'lr': 0.01,
        'batch_size': 200,
        'n_epochs': 200,
        }

        pretrained_dnn = init_DNN(best_config['size_v'])
        print('-------------Pretraining DNN-------------')
        pretrained_dnn = pretrain_DNN(pretrained_dnn,
                                    n_epochs=best_config['n_epochs_pretrain'],
                                    lr=best_config['lr_pretrain'],
                                    batch_size=best_config['batch_size_pretrain'],
                                    X=X, print_error=False)
        pretrained_dnn.rbms.append(init_RBM(200, n_classes_mnist)) # Add classification layer
        print('-------------Retropropagation Pretrained DNN-------------')
        pretrained_dnn = retropropagation(pretrained_dnn, n_epochs=best_config['n_epochs'], lr=best_config['lr'], batch_size=best_config['batch_size'], X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, plot_error=True)
        error = test_DNN(pretrained_dnn, X_valid, y_valid)
        print('Error Pretrained DNN: ', error)






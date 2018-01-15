from data_loader import load_mnist

if __name__ == '__main__':
    X_train, y_train = load_mnist('mnist', kind='train')
    
import struct
import numpy as np
import os


def read_mnist_train_images_idx3(path: str):
    bin_data = open(path, mode='rb').read()

    offset = 0
    fmt_header = '>iiii'
    _, num_image, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)

    image_size = num_rows*num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>'+str(image_size)+'B'
    images: np.ndarray = np.empty((num_image, num_rows * num_cols))
    for i in range(num_image):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    return images.astype(int)


def read_mnist_train_labels_idx1(path: str):
    bin_data = open(path, mode='rb').read()

    offset = 0
    fmt_header = '>ii'
    _, num_image = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    labels: np.ndarray = np.empty(num_image)
    fmt_label = '>B'
    for i in range(num_image):
        labels[i] = struct.unpack_from(fmt_label, bin_data, offset)[0]
        offset += struct.calcsize(fmt_label)
    return labels.astype(int)


def read_mnist(root_dir: str):
    return (read_mnist_train_images_idx3(os.path.join(root_dir, 'data/train-images.idx3-ubyte')),
            read_mnist_train_labels_idx1(os.path.join(
                root_dir, 'data/train-labels.idx1-ubyte')),
            read_mnist_train_images_idx3(os.path.join(
                root_dir, 'data/t10k-images.idx3-ubyte')),
            read_mnist_train_labels_idx1(os.path.join(root_dir, 'data/t10k-labels.idx1-ubyte')))


def __reduce_data_01(X, y):
    w = np.where((y == 0) | (y == 1))
    X = X[w]
    y = y[w]
    y[np.where(y == 0)] = -1
    return X, y


def read_mnist_01():
    train_X, train_y, test_X, test_y = read_mnist(os.getcwd())
    return __reduce_data_01(train_X, train_y), __reduce_data_01(test_X, test_y)

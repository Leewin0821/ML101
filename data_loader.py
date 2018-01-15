import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    image_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imapath:
        magic, num, rows, cols = struct.unpack('>IIII', imapath.read(16))
        images = np.fromfile(imapath, dtype=np.uint8).reshape(len(labels), 784)

    labels = labels.reshape(len(labels), 1)

    return images, labels

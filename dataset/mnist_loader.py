import numpy as np # linear algebra
import struct
from array import array
from os.path  import join


# MNIST Data Loader Class
class MnistDataloader():
    def __init__(self, test_images_filepath, test_labels_filepath):
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        images, labels = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        images = [np.array(datum).flatten().tolist() for datum in images]
        labels = labels.tolist()
        return (images, labels)   


def mnist_test_dataloader():
    # returns image / label data from mnist TEST dataset
    mnistLoader = MnistDataloader("../dataset/raw/mnist/t10k-images-idx3-ubyte", "../dataset/raw/mnist/t10k-labels-idx1-ubyte")
    return mnistLoader.load_data()

def fashion_mnist_test_dataloader():
    # returns image / label data from mnist TEST dataset
    mnistLoader = MnistDataloader("./raw_data/fmnist_test/t10k-images-idx3-ubyte", "./raw_data/fmnist_test/t10k-labels-idx1-ubyte")
    return mnistLoader.load_data()
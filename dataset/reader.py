import csv

from mnist_loader import *

## Returns raw_data, label_data
## if no label, returns None for label_data
def dataset_reader(dataset):
    return {
        "spheres": lambda : spheres_reader(),
        "mnist"  : lambda : mnist_reader(),
    }[dataset]()

def spheres_reader():    
    data = list(csv.reader(open("./raw/spheres/spheres.csv")))[1:]
    raw_data   = [[float(element) for element in  datum[:-1]] for datum in data]
    label_data = [int(float(datum[-1])) for datum in data]
    return (raw_data, label_data)

def mnist_reader():
    raw_data, label_data = mnist_test_dataloader()
    return (raw_data, label_data)
import csv

def dataset_reader(dataset):
    return {
        "spheres": lambda : spheres_reader()
    }[dataset]()

def spheres_reader():    
    data = list(csv.reader(open("./spheres/raw/spheres.csv")))[1:]
    raw_data   = [[float(element) for element in  datum[:-1]] for datum in data]
    label_data = [int(float(datum[-1])) for datum in data]
    return (raw_data, label_data)
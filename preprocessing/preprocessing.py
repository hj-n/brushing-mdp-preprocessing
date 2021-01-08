
import numpy as np
import json
import argparse
import time

from pathlib import Path
from sklearn.neighbors import KDTree

parser = argparse.ArgumentParser(description='Preprocessing Step for MDP Brushing', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", "--dataset", help='The dataset to preprocess')
parser.add_argument("-m", "--method", help='Embedding method to preprocess')
parser.add_argument("-s", "--sample", help='Sampling divisor value to preprocess')

args = parser.parse_args()

dataset        = args.dataset
method         = args.method
sample_divisor = args.sample

## parser error check
# dataset / method / sample
if (dataset == None):
    raise Exception("You should specify the dataset")
if (method == None):
    raise Exception("You should specify the method")
if (sample_divisor == None):
    raise Exception("Yout should specify the sampling divisor")
try:
    sample_divisor = int(sample_divisor)
except:
    raise Exception("Sample divisor should be an integer")


## Check whether the path is valid or not
path = "../dataset/" + dataset + "/" + method + "/" + str(sample_divisor) + "/"
if not Path(path).exists():
    raise Exception("There is no matching dataset with the path: " + path)


start = time.time()

## File reading
raw_file = open(path + "raw.json")
emb_file = open(path + "emb.json")
label_file = open(path + "label.json")

raw_data = np.array(json.load(raw_file))
emb_data = np.array(json.load(emb_file))
label_data = np.array(json.load(label_file))

length = len(raw_data)

end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to read file")


start = time.time()
## Construct KDTree and generate knn graph
k = 30
kdtree = KDTree(raw_data, leaf_size=2)
neighbors = kdtree.query(raw_data, k + 1, return_distance=False)
raw_knn = neighbors[:, 1:]

end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to generate knn graph")


start = time.time()
## Construct SNN strength matrix
snn_strength = np.zeros([length, length])

length = 500

c = 0
for i in range(length):
    for j in range(length):
        for m in range(k):
            for n in range(k):
                c += 1

end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to generate snn matrix")



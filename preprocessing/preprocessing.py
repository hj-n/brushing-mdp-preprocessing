
import numpy as np
import json
import argparse
import time
import math

from pathlib import Path
from sklearn.neighbors import KDTree

from numba import cuda

parser = argparse.ArgumentParser(description='Preprocessing Step for MDP Brushing', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", "--dataset", help='The dataset to preprocess')
parser.add_argument("-m", "--method", help='Embedding method to preprocess')
parser.add_argument("-s", "--sample", help='Sampling divisor value to preprocess')
parser.add_argument("--testgpu", action='store_true')

args = parser.parse_args()

dataset        = args.dataset
method         = args.method
sample_divisor = args.sample
is_testing = args.testgpu

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
raw_knn = np.array(neighbors[:, 1:])

end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to generate knn graph")




## GPU Acceleration for snn matrix computation

@cuda.jit
def snn(raw_knn, snn_strength):
    ## Input: raw_knn (knn info)
    ## Output: snn_strength (snn info)
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if i >= length or j >= length or j >= i:
        return

    c = 0
    for m in range(k):
        for n in range(k):
            if raw_knn[i, m] == raw_knn[j,n]:
                c += (k + 1 - m) * (k + 1 - n)

    snn_strength[i, j] = c
    


start = time.time()

raw_knn_gloabl_mem = cuda.to_device(raw_knn)
snn_strength_global_mem = cuda.device_array((length, length))

TPB = 16
tpb = (TPB, TPB)
bpg = (math.ceil(length / TPB), math.ceil(length / TPB))

snn[bpg, tpb](raw_knn_gloabl_mem, snn_strength_global_mem)
snn_strength = snn_strength_global_mem.copy_to_host()


end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to construct snn matrix with GPU")



## TESTING MODE for GPU IMPLEMENTATION (only execute when --testgpu flag is set)
if is_testing:
    start = time.time()
    ## Construct SNN strength matrix (TEST code)
    snn_strength_test = np.zeros([length, length])

    for i in range(length):
        for j in range(i):
            for m in range(k):
                for n in range(k):
                    if raw_knn[i,m] == raw_knn[j,n]:
                        snn_strength_test[i, j] += (k + 1 - m) * (k + 1 - n)

    end = time.time()
    print("Took " + "{:.3f}".format(end - start) + " seconds to construct snn matrix with cpu")

    for i in range(length):
        for j in range(i):
            if not (-0.0001 < snn_strength[i,j] - snn_strength_test[i, j] < 0.0001):
                print("*** GPU result of index " + str([i, j]) + " is wrong!! ***")
                print("GPU result: ", snn_strength[i, j])
                print("CPU result: ", snn_strength_test[i, j])
                raise Exception()







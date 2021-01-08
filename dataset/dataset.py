
import csv
import os
import json
import argparse
import reader as READ
import embedding as EMBED
import time
from pathlib import Path

SUPPORTING_DATASETS = ["spheres"]
SUPPORTING_METHODS = ["umap"]

def sampling(original_list, divisor):
    return [datum for (i, datum) in enumerate(original_list) if i % divisor == 0]

parser = argparse.ArgumentParser(description='Dataset Generation for Dimensionality Reduction Embeddings', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", "--dataset", help='The dataset to generate DR embedding\n' +
                                            'currently supports: ' + str(SUPPORTING_DATASETS))
parser.add_argument("-m", "--method", help='Embedding method which will be used\n' +
                                            'currently supports: ' + str(SUPPORTING_METHODS)
                   )
parser.add_argument("-s", "--sample", default=1, help='Sampling Divisor: sample only indices whose remainder divided by divisor is 0\n' +
                                                      'ex) --sample 3: sample out 33 percent')
parser.add_argument("-md", "--min-dist", default=0.5, help="umap argument min_dist. Only valid when you set the method as umap")
parser.add_argument("-nn", "--n-neighbors", default=60, help="umap argument n_neighbors. Only valid when you set the method as umap")

args = parser.parse_args()

dataset        = args.dataset
method         = args.method
sample_divisor = args.sample
min_dist       = args.min_dist
n_neighbors    = args.n_neighbors

## parser error check
# dataset / method
if (dataset == None):
    raise Exception("You should specify the dataset (Currently supporting " + str(SUPPORTING_DATASETS) + ")")
if (method == None):
    raise Exception("You should specify the method to use (Currently supporting " + str(SUPPORTING_METHODS) + ")")
if (dataset not in SUPPORTING_DATASETS):
    raise Exception(dataset + " dataset is not currently supported (Currently supporting " + str(SUPPORTING_DATASETS) + ")")
if (method not in SUPPORTING_METHODS):
    raise Exception(method + " method is not currently supported (Currently supporting " + str(SUPPORTING_METHODS) + ")")

# sampling
if (sample_divisor == 0):
    raise Exception("Sample divisor cannot be a 0")
try:
    sample_divisor = int(sample_divisor)
except:
    raise Exception("Sample divisor should be an integer")

# UMAP arguments
if method=="umap":
    try:
        min_dist = float(min_dist)
    except:
        raise Exception("min_dist should be a floating point")
    try:
        n_neighbors = int(n_neighbors)
    except:
        raise Exception("n_neighbors should be an integer")
    

## Start embedding data generation
print(method.upper() + " embedding for " + dataset.upper() + " dataset generation")


start = time.time()
## READ Data from raw data and sample down the dataset
raw_data, label_data = READ.dataset_reader(dataset)
raw_data = sampling(raw_data, sample_divisor)
label_data = sampling(label_data, sample_divisor) if label_data != None else label_data

end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to read file")


start = time.time()
## Generate embeddings
emb_data = EMBED.embedding(method, raw_data, min_dist, n_neighbors)

end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to generate " + method + " embedding")


## Generate path (if not exists) and dump json file
path = "./" + dataset + "/" + method + "/" + str(sample_divisor) + "/"
Path(path).mkdir(parents=True, exist_ok=True)


start = time.time()

with open(path + "raw.json", "w") as outfile:
    json.dump(raw_data, outfile)
with open(path + "emb.json", "w") as outfile:
    json.dump(emb_data, outfile)
if label_data != None:
    with open(path + "label.json", "w") as outfile:
        json.dump(label_data, outfile)

end = time.time()
print("Took " + "{:.3f}".format(end - start) + " seconds to dump result files")

print("FINISHED!!")


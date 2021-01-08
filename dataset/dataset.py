
import csv
import os
import argparse
import reader as READ

SUPPORTING_DATASETS = ["spheres"]
SUPPORTING_METHODS = ["umap"]


parser = argparse.ArgumentParser(description='Dataset Generation for Dimensionality Reduction Embeddings', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", "--dataset", help='The dataset to generate DR embedding\n' +
                                            'currently supports: ' + str(SUPPORTING_DATASETS))
parser.add_argument("-m", "--method", help='Embedding method which will be used\n' +
                                            'currently supports: ' + str(SUPPORTING_METHODS)
                   )
parser.add_argument("-s", "--sample", default=1, help='Sampling Divisor: sample only indices whose remainder divided by divisor is 0\n' +
                                                      'ex) --sample 3: sample out 33 percent')
parser.add_argument("-md", "--min-dist", help="umap argument min_dist. Only valid when you set the method as umap")
parser.add_argument("-nn", "--n-neighbors", help="umap argument n_neighbors. Only valid when you set the method as umap")

args = parser.parse_args()

dataset        = args.dataset
method         = args.method
sample_divisor = args.sample
min_dist       = args.min_dist
n_neighbors    = args.n_neighbors

## parser error check
if (dataset == None):
    raise Exception("You should specify the dataset (Currently supporting " + str(SUPPORTING_DATASETS) + ")")
if (method == None):
    raise Exception("You should specify the method to use (Currently supporting " + str(SUPPORTING_METHODS) + ")")
if (dataset not in SUPPORTING_DATASETS):
    raise Exception(dataset + " dataset is not currently supported (Currently supporting " + str(SUPPORTING_DATASETS) + ")")
if (method not in SUPPORTING_METHODS):
    raise Exception(method + " method is not currently supported (Currently supporting " + str(SUPPORTING_METHODS) + ")")


## READ Data from raw data
raw_data, label_data = READ.dataset_reader(dataset)




# Spheres data generation for final test data extraction (umap)
# spheres_data = list(csv.reader(open("./raw_data/spheres/raw.csv")))[1:]
# spheres_raw_data = np.array([datum[:-1] for datum in spheres_data])
# spheres_label = np.array([datum[-1] for datum in spheres_data])

# p = 0.3
# n = 20

# final_data = [] 
# key_summary = str(n) + "_" + str(int(p * 100))
# umap_instance = umap.UMAP(n_neighbors=n, min_dist=p)
# spheres_emb_data =umap_instance.fit_transform(spheres_raw_data)
# for (i, _) in enumerate(spheres_emb_data):
#     datum = {}
#     datum["raw"] = spheres_raw_data[i].tolist()
#     datum["emb"] = spheres_emb_data[i].tolist()
#     datum["label"] = spheres_label[i]
#     final_data.append(datum)
# print("UMAP for", "spheres", key_summary, "finished!!")
# with open(PATH + "spheres_" + key_summary + "_umap.json", "w") as outfile:
#     json.dump(final_data, outfile)
import json
from numpy import loadtxt, ndarray, min, max, array
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
import matplotlib.pyplot as plt

from SNNDPC import SNNDPC

dataset_path = "../dataset/spheres/umap/5/"

data_file = open(dataset_path + "raw.json")
emb_file = open(dataset_path + "emb.json")
label_file = open(dataset_path + "label.json")
data = array(json.load(data_file))
emb = array(json.load(emb_file))
label = array(json.load(label_file))


# for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
#     nc = 11

#     centroid, assignment = SNNDPC(k, nc, data)

#     print("k =", str(k))
#     print(f"AMI = {adjusted_mutual_info_score(label, assignment):.4f}")
#     print(f"ARI = {adjusted_rand_score(label, assignment):.4f}")
#     print(f"FMI = {fowlkes_mallows_score(label, assignment):.4f}")


#     fig, ax = plt.subplots()

#     scatter = ax.scatter(emb[:, 0], emb[:, 1], c=assignment, s=3, cmap="Spectral")
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

#     legend1 = ax.legend(*scatter.legend_elements(),
#                         loc="upper right", title="Classes", bbox_to_anchor=(1.2, 1))
#     ax.add_artist(legend1)


#     plt.savefig("./images/spheres_umap_5_" + str(k) + ".png")

fig, ax = plt.subplots()

scatter = ax.scatter(emb[:, 0], emb[:, 1], c=label, s=3, cmap="Spectral")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Classes", bbox_to_anchor=(1.2, 1))
ax.add_artist(legend1)


plt.savefig("./images/spheres_umap_5_real.png")


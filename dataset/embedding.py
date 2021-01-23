import umap
from sklearn.decomposition import PCA

def embedding(method, raw_data, min_dist, n_neighbors):
    return {
        "umap": lambda : umap_embedding(raw_data, min_dist, n_neighbors),
        "pca" : lambda : pca_embedding(raw_data)
    }[method]()

def umap_embedding(raw_data, min_dist, n_neighbors):
    umap_instance = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    emb_data = umap_instance.fit_transform(raw_data)
    emb_data = emb_data.tolist()
    return emb_data

def pca_embedding(raw_data):
    pca =  PCA(n_components=2)
    emb_data = pca.fit_transform(raw_data).tolist()
    return emb_data
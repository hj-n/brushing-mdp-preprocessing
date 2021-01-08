import umap

def embedding(method, raw_data, min_dist, n_neighbors):
    return {
        "umap": lambda : umap_embedding(raw_data, min_dist, n_neighbors)
    }[method]()

def umap_embedding(raw_data, min_dist, n_neighbors):
    umap_instance = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    emb_data = umap_instance.fit_transform(raw_data)
    emb_data = emb_data.tolist()
    return emb_data
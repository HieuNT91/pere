import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

def get_k_popular_items(inbound_ids, outbound_ids, n_popular, k=20):
    topk_ids = []
    for x in sorted(inbound_ids):
        topk_ids.append(x)
    if len(topk_ids) < k:
        for x in sorted(outbound_ids):
            if x < n_popular:
                topk_ids.append(x)
    return topk_ids[:k]

def get_k_closest_items(item_embeddings, user, k=20):
    if len(user.shape) > 1:
        user = np.squeeze(user)
    distances = euclidean_distances(item_embeddings, user[np.newaxis,:], squared=True)
    topk_ids = np.argsort(distances, axis=0)[:k]
    return topk_ids

def find_plane_coordinate(x1, x2):
    """Find Wx + b = 0 orthogonal to vector by x1, x2
    x1, x2 has dimension [1, embedding_dim]
    """
    W = (x1 - x2)
    x_mid = (x1 + x2) / 2
    b = -W@x_mid.T
    return (W, b)

def get_all_boundary(rankings):
    """
    """
    # Get 2-element combinations of center_ids
    boundaries = []
    for center, pref_center in rankings:
        plane_coordinates = find_plane_coordinate(center, pref_center)
        boundaries.append(plane_coordinates)

    return boundaries

def get_inbound_ids_(plane_coordinates, x_pref, item_embeddings):
    """ Get ids of items that lie on the same side as preferrence item
    plane_coordinates: 
        W [1, embedding_dim]
        b [scalar]
    x_pref: item embedding prefered by user. [1, embedding_dim]
    all_item: all item embedding. [n_samples, embedding_dim]
    """
    W, b = plane_coordinates
    y = W@x_pref.T + b
    y_all = W@item_embeddings.T + b
    
    # Get ids of y_all that have same sign of y
    mul = np.multiply(y_all, y)
    return np.argwhere(mul > 0)

def get_inbound_ids(rankings, item_embeddings):
    output = []
    for center, pref_center in rankings:
        plane_coordinates = find_plane_coordinate(center, pref_center)
        inbound_ids = get_inbound_ids_(plane_coordinates, pref_center, item_embeddings)
        
        item_ids = []
        for item_id in list(inbound_ids):
            item_ids.append(item_id[0])
        output.append(item_ids)

    # Take intersection of all sets
    final_set = set(output[0]).intersection(*output[1:])
    return list(final_set)


def load_data(data, n_samples=100, load_all=True):
    with open(f'embeddings/{data}_item_embedding.npy', 'rb') as f:
        item_embedding = np.load(f)

    with open(f'embeddings/{data}_item_popularity.npy', 'rb') as f:
        item_popularity = np.load(f)

    with open(f'embeddings/{data}_user_embedding.npy', 'rb') as f:
        user_embedding = np.load(f)
    
    if load_all == True:
        return item_embedding, item_popularity, user_embedding
    else:
        return item_embedding[:n_samples], item_popularity[:n_samples], user_embedding[:n_samples]

def get_embedding_pairs_from_id_pairs(id_pairs, item_embeddings):
    embeddings = []
    for pair in id_pairs:
        embedding_pair = (item_embeddings[pair[0]], item_embeddings[pair[1]])
        embeddings.append(embedding_pair)
    return embeddings

def accuracy(pred, gt):
    hit = 0
    for p in pred:
        if p in gt:
            hit += 1 
    return hit / len(gt)

def calculate_accuracy(user, n_popular=None, item_embeddings=None, chebyshev=None, 
                    inbound_ids=None, outbound_ids=None, 
                    rec='center', k=20):
    
    gt = get_k_closest_items(item_embeddings, user, k=k)
    if rec == 'center' or rec == 'ask_popular':
        pred = get_k_closest_items(item_embeddings, chebyshev, k=k)
    elif rec == 'rec_popular':
        pred = get_k_popular_items(inbound_ids, outbound_ids, n_popular=n_popular, k=k)
    return accuracy(pred, gt)

def normalize_old(arr):
    return (arr - arr.min(axis=0, keepdims=True)) / (arr.max(axis=0, keepdims=True) - arr.min(axis=0, keepdims=True))

def normalize(user_embeddings, item_embeddings):
    arr = np.concatenate((user_embeddings, item_embeddings), axis=0)
    normalized_arr = (arr - arr.min(axis=0, keepdims=True)) / (arr.max(axis=0, keepdims=True) - arr.min(axis=0, keepdims=True))
    return normalized_arr[:user_embeddings.shape[0]], normalized_arr[user_embeddings.shape[0]:]

def save_user_info(path, user_data, dname):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, f'{dname}_users.pickle'), 'wb') as handle:
        pickle.dump(user_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_user_info(path, dname):
    with open(os.path.join(path, f'{dname}_users.pickle'), 'rb') as handle:
        user_data = pickle.load(handle)

    return user_data

def save_numpy(path, top100, dname, seed):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, f'{dname}_rec_{seed}.npy'), 'wb') as handle:
        np.save(handle, top100)


def load_numpy(path, dname, seed):
    with open(os.path.join(path, f'{dname}_rec_{seed}.npy'), 'rb') as handle:
        top100 = np.load(handle)

    return top100
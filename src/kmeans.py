import warnings
from copy import deepcopy as copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum

###### Config
SEED = 1
N_CLUSTERS = 40
CLUSTER_COLORS = ['darkgreen', 'darkblue', 'darkkhaki', 'darkmagenta', 'darkred']
CENTER_COLORS = ['lightgreen', 'lightblue', 'khaki', 'magenta', 'crimson']


# adapt from sklearn.clustering.kmeans_plusplus
def kmeans_plusplus(X, n_clusters, x_squared_norms, random_state):
    n_samples, n_features = X.shape 
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Trick in Kmeans++ by Arthur/Vassilvitskii
    n_local_trials = 2 + int(np.log(n_clusters))
    center_id = random_state.randint(n_samples)
    ids = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    ids[0] = center_id
    
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()
    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        ids[c] = best_candidate
    return centers, ids

def calc_inertia(X, centers, labels):
    inertia = 0
    for j in range(centers.shape[0]):
        inertia += euclidean_distances(X[labels==j, :], centers[j, np.newaxis], squared=True).sum()
    return inertia


def _is_same_clustering(labels1, labels2, n_clusters):
    """Check if two arrays of labels are the same up to a permutation of the labels"""
    mapping = np.full(fill_value=-1, shape=(n_clusters,), dtype=np.int32)

    for i in range(labels1.shape[0]):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False
    return True


def visualize_kmeans(X, center_inds, label, epoch=1, sample_weight=None):
    fig = go.Figure()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    sample_weight_rescaled = None 
    if sample_weight is None: 
        size = 5
    else:
        sample_weight_rescaled = np.interp(sample_weight, (sample_weight.min(), sample_weight.max()), 
                                          (5, 18))
        # print(sample_weight_rescaled)
    # components = X.copy()
    cluster_sizes = get_cluster_sizes(label)

    for i in range(len(center_inds)):
        emb = components[label==i]
        
        if sample_weight is not None:
            sizes = sample_weight_rescaled[label==i]

        for j in range(emb.shape[0]):
            x = [emb[j, 0]]
            y = [emb[j, 1]]
            if sample_weight is not None:
                size = sizes[j]
            showlegend = False
            if j == 0:
                showlegend = True

            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=x,
                    y=y,
                    opacity=0.8,
                    name=f'Cluster {i} - {cluster_sizes[i]} pts',
                    marker=dict(
                        color=CLUSTER_COLORS[i],
                        size=size
                    ),
                    showlegend=showlegend
                )
            )

        x, y = [components[center_inds[i], 0]], [components[center_inds[i], 1]]

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=x,
                y=y,
                opacity=1,
                marker_symbol='star',
                name=f'center {i}',
                marker=dict(
                    color=CENTER_COLORS[i],
                    size=20
                ),
                showlegend=True
            )
        )
    fig.write_image(f'kmeans_{epoch}.png')


def kmeans_single_run(X,
    centers_init,
    n_popular_items,
    sample_weight=None,
    max_iter=300,
    verbose=False,
    tol=1e-4, 
    centers_ids=None):

    n_clusters = centers_init.shape[0]
    centers = centers_init.copy()
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    strict_convergence = False
    labels_old = np.full(X.shape[0], -1, dtype=np.int32)
    for i in range(max_iter):
        
        distances = euclidean_distances(X, centers, squared=True)
        labels = np.argsort(distances, axis=1)[:, 0]
        
        # visualize_kmeans(X, centers_ids, labels, i, sample_weight=sample_weight)
        assert len(set(labels)) == n_clusters, f'labels set: {set(labels)}'
        
        new_center_ids = []
        for j in range(n_clusters):
            popular_labels = labels[:n_popular_items]
            d = euclidean_distances(X[labels==j], X[:n_popular_items, :][popular_labels==j], squared=True)
            if sample_weight is not None: 
                d = np.multiply(d, sample_weight[labels==j][:, np.newaxis])
            sum_distances = d.sum(axis=0)

            extended_sum_distances = []
            for ii in range(len(popular_labels)):
                if popular_labels[ii] == j:
                    extended_sum_distances.append(sum_distances[0])
                    sum_distances = np.delete(sum_distances, 0)
                else: 
                    extended_sum_distances.append(np.inf)

            extended_sum_distances = np.array(extended_sum_distances)
            assert extended_sum_distances.size == n_popular_items
            new_center_ids.append(np.argmin(extended_sum_distances))

        
        new_centers = X[new_center_ids, :]
        # visualize_kmeans(X, new_center_ids, labels, i+0.5, sample_weight=sample_weight)
        if len(new_center_ids) > len(set(new_center_ids)):
            breakpoint()

        if verbose:
            inertia = calc_inertia(X, centers, labels)
            print(f"Iteration {i}, inertia {inertia}.")

        if np.array_equal(labels, labels_old):
            if verbose:
                print(f"Converged at iteration {i}: strict convergence")
            strict_convergence = True
            break
        else:
            for j in range(n_clusters):
                center_shift[j] = euclidean_distances(centers[j, np.newaxis], new_centers[j, np.newaxis])[0]
            center_shift_tot = (center_shift ** 2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                distances = euclidean_distances(X, new_centers, squared=True)
                labels = np.argsort(distances, axis=1)[:, 0]
                break

        labels_old[:] = copy(labels)
        centers = copy(new_centers)
        centers_ids = copy(new_center_ids)
        centers_unique = np.unique(centers, axis=0)

        if centers_unique.shape[0] < centers.shape[0]:
            centers = copy(centers_unique) 
            
        n_clusters = centers.shape[0]
    
    inertia = calc_inertia(X, centers, labels)
    return labels, inertia, new_centers, new_center_ids, i+1

def kmeans(X, n_init, n_clusters, n_popular, sample_weight):
    """ Find k centroids
    
    X: data [n_samples, n_features]
    """
    
    x_squared_norms = row_norms(X, squared=True)
    # X_mean = X.mean(axis=0)
    # X -= X_mean

    best_inertia, best_labels = None, None
    for i in range(n_init):
        random_state = np.random.RandomState(i+15)
        np.random.seed(i+10)
        centers_init, ids = kmeans_plusplus(X[:n_popular, :], n_clusters, 
                                       x_squared_norms[:n_popular], 
                                       random_state)
        # print("Initialization complete")

        labels, inertia, centers, center_ids, n_iter_ = kmeans_single_run(
            X, centers_init, n_popular, sample_weight=sample_weight, max_iter=300,
            verbose=False, tol=1e-4, centers_ids=ids
        )

        if best_inertia is None or (
            inertia < best_inertia
            and not _is_same_clustering(labels, best_labels, n_clusters)
        ):
            best_labels = labels
            best_centers = centers
            best_inertia = inertia
            best_center_ids = center_ids
            best_n_iter = n_iter_
    
    distinct_clusters = len(set(best_labels))
    if distinct_clusters < n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, n_clusters)
            )

    output = {'best_centers': best_centers,
              'best_labels': best_labels,
              'best_inertia': best_inertia,
              'best_center_ids': best_center_ids,
              'best_n_iter': best_n_iter}

    return output


def get_cluster_sizes(labels):
    labels_dict = {}
    for label in labels:
        labels_dict[label] = labels_dict.get(label, 0) + 1
    return labels_dict

if __name__ == '__main__':

    
    with open('embedding/gowalla_item_embedding.npy', 'rb') as f:
        item_embedding = np.load(f)

    with open('embedding/gowalla_item_popularity.npy', 'rb') as f:
        item_popularity = np.load(f)

    n_samples = 3000
    n_popular = 200
    result = kmeans(item_embedding[:n_samples], n_init=1, n_clusters=N_CLUSTERS, 
                        n_popular=n_popular, sample_weight=item_popularity[:n_samples])

    cluster_sizes = get_cluster_sizes(result['best_labels'])
    for k, v in cluster_sizes.items():
        print('cluster ', k, ' has ', v, ' items')
    
    print('Ranking of centers:', result['best_center_ids'])
    print('popularity of centers:', item_popularity[result['best_center_ids']])

    # visualize_kmeans(item_embedding[:n_samples], result['best_center_ids'], 
    #                 result['best_labels'], epoch=99,
    #                 sample_weight=item_popularity[:n_samples, :])



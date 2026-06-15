import numpy as np
from scipy.stats import bernoulli
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from .utils import *
import plotly.express as px
import itertools 
import tqdm
from scipy.special import expit

def plot_kappa(user_embeddings, item_embeddings, item_popularity, max_kappa=50, num_favorite_item=200):
    """
    return new_user, probs that user watch item i
    """
    num_item = item_embeddings.shape[0]
    feature_dim = item_embeddings.shape[1]
    mean = np.mean(user_embeddings, axis=0)
    cov = np.cov(user_embeddings.T)
    assert cov.shape[0] == mean.shape[0]
    new_user = np.clip(np.random.multivariate_normal(mean, cov, size=1), a_min=0, a_max=1)
    

    x = np.arange(0.1, np.sqrt(feature_dim), 0.1)
    item_id = 100
    for kappa_0 in [1, 2, 3, 4, 5, 6, 7]:
        y = []
        # y_ = []
        for x_val in x:
            a = kappa_0 / (np.sqrt(feature_dim) - x_val)
            b = 1 / x_val
            logit = sigmoid(b - a)
            y.append(item_popularity[item_id] * logit)
        plt.plot(x, y, label=f'kappa={kappa_0}')
        plt.title(f'probability term vs c0')


    plt.legend()
    plt.xticks([0, np.sqrt(feature_dim)])
    plt.yticks([0, item_popularity[item_id], 1])
    plt.xlabel("C0")
    plt.ylabel("Experience Probability")
    plt.show()


def gen_user(user_embeddings, item_embeddings, item_popularity, min_kappa=17, max_kappa=21, num_favorite_item=200, num_users=10, seed=1, dname=None):
    """
    return new_user, probs that user watch item i
    """
    common_dir = f"data/{dname}/new_user/minmaxkappa_{min_kappa}{max_kappa}_seed_{seed}"
    file_name = os.path.join(common_dir, f"{dname}_users.pickle")
    if os.path.exists(file_name):
        user_info = load_user_info(common_dir, dname)
        if user_info['is_watcheds'].shape[0] >= num_users:
            return user_info

    np.random.seed(seed)
    num_items = item_embeddings.shape[0]
    fav = None
    watch = None

    for i in tqdm.tqdm(range(num_users)):
        favorite_items = get_k_closest_items(item_embeddings, user_embeddings[i], k=num_favorite_item)
        if fav is None:
            fav = favorite_items.reshape(1, -1)
        else:
            fav = np.concatenate((fav, favorite_items.reshape(1, -1)), axis=0)
        kappa_0 = np.random.randint(min_kappa, max_kappa+1, size=num_items)
        is_watched = []
        
        feature_dim = item_embeddings.shape[0]
        c0 = np.linalg.norm(user_embeddings[i] - item_embeddings, axis=1)
        a = 1 / c0
        b = kappa_0 / (np.sqrt(feature_dim) - c0)
        probs = item_popularity * expit(a-b)

        for j in range(item_embeddings.shape[0]):
            is_watched.append(bernoulli.rvs(size=1,p=probs[j])[0])

        if watch is None:
            watch = np.array(is_watched).reshape(1, -1)
        else:
            watch = np.concatenate((watch, np.array(is_watched).reshape(1, -1)), axis=0)

    user_info = {}
    user_info['new_users'] = user_embeddings[:num_users]
    user_info['is_watcheds'] = watch
    user_info['favorite_item_ids'] = fav
    save_user_info(common_dir, user_info, dname)    
    return user_info


# def gen_new_user(user_embeddings, item_embeddings, item_popularity, min_kappa=17, max_kappa=21, num_favorite_item=200, num_users=10, seed=1, dname=None):
#     """
#     return new_user, probs that user watch item i
#     """
#     common_dir = f"data/{dname}/new_user/minmaxkappa_{min_kappa}{max_kappa}_seed_{seed}"
#     file_name = os.path.join(common_dir, f"{dname}_users.pickle")
#     if os.path.exists(file_name):
#         user_info = load_user_info(common_dir, dname)
#         if user_info['new_users'].shape[0] >= num_users:
#             return user_info

#     np.random.seed(seed)
#     mean = np.mean(user_embeddings, axis=0)
#     cov = np.cov(user_embeddings.T)
#     assert cov.shape[0] == mean.shape[0]
#     new_users = np.random.multivariate_normal(mean, cov, size=num_users)
#     num_items = item_embeddings.shape[0]
#     fav = None
#     watch = None

#     for i in tqdm.tqdm(range(num_users)):
#         favorite_items = get_k_closest_items(item_embeddings, new_users[i], k=num_favorite_item)
#         if fav is None:
#             fav = favorite_items.reshape(1, -1)
#         else:
#             fav = np.concatenate((fav, favorite_items.reshape(1, -1)), axis=0)
#         kappa_0 = np.random.randint(min_kappa, max_kappa+1, size=num_items)
#         is_watched = []
        
#         feature_dim = item_embeddings.shape[0]
#         c0 = np.linalg.norm(new_users[i] - item_embeddings, axis=1)
#         a = 1 / c0
#         b = kappa_0 / (np.sqrt(feature_dim) - c0)
#         probs = item_popularity * expit(a-b)

#         for j in range(item_embeddings.shape[0]):
#             is_watched.append(bernoulli.rvs(size=1,p=probs[j])[0])

#         if watch is None:
#             watch = np.array(is_watched).reshape(1, -1)
#         else:
#             watch = np.concatenate((watch, np.array(is_watched).reshape(1, -1)), axis=0)

#     user_info = {}
#     user_info['new_users'] = new_users
#     user_info['is_watcheds'] = watch
#     user_info['favorite_item_ids'] = fav
#     save_user_info(common_dir, user_info, dname)    
#     return new_users

def plot_prob(item_embedding, user):
    logits = []
    for x in list(np.arange(20)):
        c0 = np.linalg.norm(user - item_embedding)
        # print(item_embedding.shape)
        a = x / (np.sqrt(item_embedding.shape[1]) - c0)
        b = 1 / c0
        logit = sigmoid(b - a)
        logits.append(logit)
    plt.plot(logits)
    plt.ylabel('Sigmoid')
    plt.xlabel('Kappa_0')
    plt.show()

if __name__ == '__main__':
    dname = 'gowalla'
    # from user_interaction import normalize
    item_embeddings, item_popularity, user_embeddings = load_data(data=dname, load_all=True)
    num_samples = min(len(item_popularity), item_embeddings.shape[0])
    item_embeddings = normalize(item_embeddings)
    
    # user_embeddings = normalize(user_embeddings)
    item_popularity = normalize(item_popularity)
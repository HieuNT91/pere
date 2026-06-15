import random 
import numpy as np
from .user_interaction import * 
from .utils import *
from .chebysev import * 
from .kmeans import *
from .point2convex import *
from sklearn.metrics import ndcg_score
from scipy.special import expit
from .dpp import map_inference_dpp_local_search_2
from scipy.linalg import lu
from scipy.sparse.linalg import svds

def get_pairs(liked_list, disliked_list):
    id_pairs = []
    for disliked_item_id in disliked_list:
        for liked_item_id in liked_list:
            id_pairs.append((disliked_item_id, liked_item_id))
    return id_pairs

class RandomSeedItem:
    """ Step 1: Random
    """
    def __init__(self, num_popular_items):
        self.num_popular_items = num_popular_items

    def pick(self, K=10):
        """ return ids of popular item
        """
        ids = []
        population = list(np.arange(self.num_popular_items))
        for _ in range(K):
            
            choice = random.choices(population, k=1)[0]
            ids.append(choice)
        return ids



class DPE:
    """ https://dl.acm.org/doi/pdf/10.1145/3437963.3441786 
    """
    def __init__(self, user_info, train_rating_dict, category_to_items):
        # train_rating = n_users x n_items 
        self.train_rating_dict = train_rating_dict
        self.item_to_userset = {}
        for user, items in self.train_rating_dict.items():
            for item in items: 
                if item not in self.item_to_userset:
                    self.item_to_userset[item] = [user]
                else:
                    self.item_to_userset[item].append(user)

        self.user_num = len(self.train_rating_dict)
        self.user_info = user_info
        self.category_to_items = category_to_items
        self.arms_num = len(category_to_items)
        self.normal_paramteres = []
        self.categories = list(category_to_items.keys())
        for i in range(self.arms_num):
            self.normal_paramteres.append([0, 0])


    def IL_dist_old(self, new_idx, item_idxs):
        dist = 0
        for idx in item_idxs:
            sim = 0
            sim_i = 0
            sim_j = 0
            for u in range(self.user_num):
                r_i = 1 if new_idx in self.train_rating_dict[u] else 0
                r_j = 1 if idx in self.train_rating_dict[u] else 0
                sim += r_i * r_j 
                sim_i += r_i ** 2 
                sim_j += r_j ** 2 
            dist += 1 - sim / (np.sqrt(sim_i) * np.sqrt(sim_j))
        return (1 / len(item_idxs)) * dist

    def IL_dist(self, new_idx, item_idxs):
        dist = 0
        userset_j = set(self.item_to_userset[new_idx])
        for idx in item_idxs:
            userset_i = set(self.item_to_userset[idx])
            numerator = len(userset_i & userset_j) 
            dist += 1 - numerator / (np.sqrt(len(userset_i)) * np.sqrt(len(userset_j)))
        return (1 / len(item_idxs)) * dist

    def pick(self, K=10):
        chosen_idxs = [] 
        for t in range(K):
            thetas = []
            for i in range(self.arms_num):
                mu, k = self.normal_paramteres[i]
                sigma = 1 / (k + 1)
                thetas.append(np.random.normal(mu, sigma))
            next_category_idx = np.argmax(thetas)
            category = self.categories[next_category_idx]
            category_item_list = self.category_to_items[category]
            if len(chosen_idxs) == 0:
                category_scores = [1] * len(category_item_list)
            else:
                category_scores = []

                for item in category_item_list:
                    category_scores.append(self.IL_dist(item, chosen_idxs))
            next_item_idxs = np.argsort(category_scores)[::-1]
            next_item = None
            for next_item_idx in list(next_item_idxs):
                next_item = self.category_to_items[category][next_item_idx]
                if next_item not in chosen_idxs:
                    break
            r = 0 
            if next_item in self.user_info['favorite_item_list']:
                r = 1
            new_mu = (mu * k + r) / (k + 2)
            new_k = k + 1 
            self.normal_paramteres[next_category_idx][0] = new_mu
            self.normal_paramteres[next_category_idx][1] = new_k 
            chosen_idxs.append(next_item)

        return chosen_idxs

class RecMaxVolSeedItem:
    def __init__(self, rating_matrix, decompose_rank=5):
        self.decompose_rank = decompose_rank
        self.R = rating_matrix
    
    def pick(self, K=10):
        """ return ids of popular item
        """
        _, _, Q = svds(self.R, k=self.decompose_rank)
        k, _, _ = lu(Q.T, p_indices=True)
        k = k[:self.decompose_rank]
        S = Q[:, k]
        S_inv = np.linalg.inv(S)
        C = S_inv @ Q 
        norms = []
        for i in range(C.shape[1]):
            norms.append(np.linalg.norm(C[:, i]))
        
        idxs = k.tolist()
        while len(idxs) < K:
            norms_idxs = np.argsort(norms)[::-1]
            idx = None 
            for i in range(len(norms_idxs)):
                if norms_idxs[i] not in idxs:
                    idx = norms_idxs[i]
                    break 
            idxs.append(idx)
            ci = C[:, idx]
            upper_C = C - (ci@ci.T * C) / (1 + ci.T@ci)
            lower_C = ci.T @ C / (1 + ci.T@ci)
            lower_C = lower_C.reshape(1, -1)
            C = np.concatenate((upper_C, lower_C), axis=0)
            ci = C[:, idx]
            for j in range(len(norms)):
                norms[j] -= (ci.T @ C[:, j])**2 / (1 + ci.T @ ci)
        return idxs
    
class GreedySeedItem:
    """ Step 1: Random
    """
    def __init__(self, num_popular_items, weights):
        self.num_popular_items = num_popular_items
        self.weights = weights

    def pick(self, K=10):
        """ return ids of popular item
        """

        return list(np.argsort(self.weights)[::-1][:K])

class KmedoidSeedItem:
    """ Step 1: Random
    """
    def __init__(self, item_embeddings, n_init, num_popular_items, weights):
        self.item_embeddings = item_embeddings 
        self.n_init = n_init 
        self.num_popular_items = num_popular_items
        self.weights = weights

    def pick(self, K=10):
        """ return ids of popular item
        """
        result = kmeans(self.item_embeddings, n_init=self.n_init, n_clusters=K, 
                    n_popular=self.num_popular_items, sample_weight=self.weights)
        return result['best_center_ids']

class DPPSeedItem:
    def __init__(self, item_embeddings, num_popular_items, weights):
        self.item_embeddings = item_embeddings 
        self.num_popular_items = num_popular_items
        self.weights = weights
        
    def pick(self, K=10, gamma=0.5):
        """ return ids of popular item
        """
        S = self.item_embeddings[:self.num_popular_items] @ self.item_embeddings[:self.num_popular_items].T
        D = self.weights[:self.num_popular_items] * np.identity(self.num_popular_items)
        L = (gamma * S + (1 - gamma) * D) * 10
        # L = self.item_embeddings[:self.num_popular_items]@self.item_embeddings[:self.num_popular_items].T
        slt_idx, _, _, _, _, _  = map_inference_dpp_local_search_2(L, k=K)
        return list(slt_idx)


class BinPopularitySeedItem:
    def __init__(self, num_popular_items, weights):
        self.num_popular_items = num_popular_items
        self.weights = weights
    
    def pick(self, K=10):
        """ return ids of popular item
        """
        def divide_chunks(arr, bin_size=3):
            for i in range(0, len(arr), bin_size):
                yield arr[i:i + bin_size]

        bin_weights = divide_chunks(self.weights[:self.num_popular_items], bin_size=self.num_popular_items * 8 // K)
        bin_ids = divide_chunks(np.arange(self.num_popular_items), bin_size=self.num_popular_items * 8 // K)
        output_ids = []
        for bin_weight, bin_id in zip(bin_weights, bin_ids):
            sorted_ids = np.argsort(bin_weight)[::-1]
            output_ids += list(bin_id[sorted_ids][:3])

        return output_ids[:K]

class BaseSequential:
    def __init__(self, item_embeddings, max_questions, num_popular_items, user_info, item_popularity, num_items_per_question, save=False, logger=None):
        self.max_questions = max_questions 
        # self.seed_item_ids = seed_item_ids
        self.num_popular_items = num_popular_items
        self.user_info = user_info
        self.item_embeddings = item_embeddings
        self.item_popularity = item_popularity
        self.feature_dim = item_embeddings.shape[1]
        self.name = "Base"
        self.num_items_per_question = num_items_per_question
        self.centers = []
        self.gt = list(get_k_closest_items(self.item_embeddings, self.user_info['embedding'], k=80))
        self.gt = [x[0] for x in self.gt[:80]]
        self.preds = []
        self.pairs = []
        self.gts = []
        self.save = save
        self.logger = logger


    def ask_user_to_rate(self):
        """ obtain L+, L- and LNA
        """    
        liked_list, disliked_list, NA_list = [], [], []
        for iid in self.seed_item_ids:
            if self.user_info['is_watched'][iid] == 0:
                NA_list.append(iid)
            else:
                if iid in self.user_info['favorite_item_list']:
                    liked_list.append(iid)
                else:
                    disliked_list.append(iid)

        response = {'liked_list': liked_list, 'disliked_list': disliked_list, 'NA_list':NA_list}
        return response


    def ask_initial_seed_items(self):
        self.user_response = self.ask_user_to_rate()
        self.pairs = get_pairs(self.user_response['liked_list'], self.user_response['disliked_list'])
        embedding_pairs = get_embedding_pairs_from_id_pairs(id_pairs=self.pairs, item_embeddings=self.item_embeddings)
        _, self.center = chebysev_center(d=self.feature_dim, U=embedding_pairs, epsilon=1e-3)



    def ask_and_update(self, seed_item_ids):
        self.seed_item_ids = seed_item_ids
        self.user_response = self.ask_user_to_rate()
        if len(self.user_response['liked_list']) > 0 and len(self.user_response['disliked_list']) > 0:
            # if len(self.user_response['liked_list']) + len(self.user_response['disliked_list']) > 0:
            self.pairs += get_pairs(self.user_response['liked_list'], self.user_response['disliked_list'])
            self.pairs = list(set(self.pairs))
            #print(f"{len(self.pairs)} pair founds, empty liked list, {len(self.user_response['liked_list'])}, {len(self.user_response['disliked_list'])}")
            embedding_pairs = get_embedding_pairs_from_id_pairs(id_pairs=self.pairs, item_embeddings=self.item_embeddings)
            #breakpoint()
            _, self.center = chebysev_center(d=self.feature_dim, U=embedding_pairs, epsilon=1e-3)
        else:
            #print(f"None pair found, empty liked list, {len(self.user_response['liked_list'])}, {len(self.user_response['disliked_list'])}")
            self.center = np.zeros(self.feature_dim) + 0.5
        self.centers.append(self.center.copy())


    def estimate_prob_old(self):
        feature_dim = self.center.shape[0]
        c0 = np.linalg.norm(self.center - self.item_embeddings, axis=1)
        a = 1 / c0
        b = 1 / (np.sqrt(feature_dim) - c0)
        probs = self.item_popularity * expit(a-b)
        return probs.tolist()

    def estimate_prob(self):
        feature_dim = self.center.shape[0]
        c0 = np.linalg.norm(self.center - self.item_embeddings, axis=1)
        a = 1 / c0
        b = 1 / (np.sqrt(feature_dim) - c0)
        probs = self.item_popularity * np.tanh(a-b)
        return probs.tolist()

    def recommend_center(self, K=20):
        self.initial_seed_items_num = len(self.seed_item_ids) - self.num_items_per_question * (len(self.centers) - 1)
        c = np.average(np.array(self.centers), weights=[self.initial_seed_items_num]+[self.num_items_per_question]*(len(self.centers)-1), axis=0)
        return get_k_closest_items(self.item_embeddings, c, k=K)
    
    def recommend_liked_centroid(self, K=20):
        embedding_pairs = get_embedding_pairs_from_id_pairs(id_pairs=self.pairs, item_embeddings=self.item_embeddings)
        inbound_ids = get_inbound_ids(embedding_pairs, self.item_embeddings)
        
        liked_ids = list(set(inbound_ids) & set(self.user_response['liked_list']))
        if len(liked_ids) > 0:
            centroid = np.average(self.item_embeddings[liked_ids], axis=0)
        else:
            min_dist = np.inf 
            min_id = 0
            for iid in self.user_response['liked_list']:
                dist = item2set(self.feature_dim, embedding_pairs, self.item_embeddings[iid])
                if dist < min_dist: 
                    min_dist = dist 
                    min_id = iid
            centroid = self.item_embeddings[min_id]

        return get_k_closest_items(self.item_embeddings, centroid, k=K)

    def recommend_popularity(self, K=20):
        embedding_pairs = get_embedding_pairs_from_id_pairs(id_pairs=self.pairs, item_embeddings=self.item_embeddings)
        inbound_ids = get_inbound_ids(embedding_pairs, self.item_embeddings)
        self.logger.info(f"There is {len(inbound_ids)} left in U")
        return sorted(inbound_ids)[:K]


class RandomSequential(BaseSequential):
    """ Step 2: Random
    """
    def __init__(self, item_embeddings, max_questions, num_popular_items, user_info, item_popularity, num_items_per_question, save=False, logger=None):
        super().__init__(item_embeddings=item_embeddings, max_questions=max_questions, num_popular_items=num_popular_items, user_info=user_info, item_popularity=item_popularity, num_items_per_question=num_items_per_question, save=save, logger=logger)
        self.name = "Random"

    def find_next_items(self, num_items):
        """ return id of next item
        """
        weights = np.ones(self.num_popular_items)
        for iid in self.seed_item_ids:
            weights[iid] = 0
        population = list(np.arange(self.num_popular_items))
        choices = random.choices(population, weights=weights[:self.num_popular_items], k=num_items)
        return list(choices)
    
    # def recommend_center(self, K=20):
    #     return get_k_closest_items(self.item_embeddings, self.center, k=K)

    
class PersonalizedSequential(BaseSequential):
    """ Step 2: Ours
    """
    def __init__(self, item_embeddings, max_questions, num_popular_items, user_info, item_popularity, num_items_per_question, save=False, logger=None, mode='avg'):
        super().__init__(item_embeddings=item_embeddings, max_questions=max_questions, num_popular_items=num_popular_items, user_info=user_info, item_popularity=item_popularity, num_items_per_question=num_items_per_question, save=save, logger=logger)
        self.name = "Personalized"
        self.mode = mode
    
    
    def find_next_items(self, num_items=5):
        weights = self.estimate_prob()
        # weights = self.item_popularity
        for iid in self.seed_item_ids:
            weights[iid] = 0
        
        max_like = 0
        min_dislike = np.inf

        if len(self.user_response['liked_list']) == 0 or len(self.user_response['disliked_list']) == 0:
            choices = np.argsort(weights)[::-1][:num_items]
            return list(choices)

        for item_id in self.user_response['liked_list']: 
            # print(chebyshev_center[:, np.newaxis].shape, item_embeddings[item_id].shape)
            dist = euclidean_distances(self.center.reshape(1, -1), self.item_embeddings[item_id].reshape(1, -1))[0][0]
            # print(dist)
            if dist > max_like:
                max_like = dist 
        
        for item_id in self.user_response['disliked_list']: 
            dist = euclidean_distances(self.center.reshape(1, -1), self.item_embeddings[item_id].reshape(1, -1))[0][0]
            if dist < min_dislike:
                min_dislike = dist
        
        vals, ids = [], []
        
        for i in range(self.num_popular_items):
            if i not in self.user_response['liked_list'] + self.user_response['disliked_list'] + self.user_response['NA_list']:
                dist_a = euclidean_distances(self.center.reshape(1, -1), self.item_embeddings[i].reshape(1, -1))[0][0]
                if dist_a <= max_like:
                    dist = chebyshev_to_K_dist(self.center, self.item_embeddings[self.user_response['liked_list']], self.item_embeddings[i], mode=self.mode, liked=True)
                elif dist_a >= min_dislike:
                    dist = chebyshev_to_K_dist(self.center, self.item_embeddings[self.user_response['disliked_list']], self.item_embeddings[i], mode=self.mode, liked=False)
                else:
                    dist = min(chebyshev_to_K_dist(self.center, self.item_embeddings[self.user_response['liked_list']], self.item_embeddings[i], mode=self.mode, liked=True),
                                chebyshev_to_K_dist(self.center, self.item_embeddings[self.user_response['disliked_list']], self.item_embeddings[i], mode=self.mode, liked=False))
                vals.append( (1-weights[i]) * dist)
                ids.append(i)
        
        sorted_ids = np.argsort(vals)[:num_items]
        return list(np.array(ids)[sorted_ids])

class GreedySequential(BaseSequential):
    """ Step 2: Ours
    """
    def __init__(self, item_embeddings, max_questions, num_popular_items, user_info, item_popularity, num_items_per_question, save=False, logger=None):
        super().__init__(item_embeddings=item_embeddings, max_questions=max_questions, num_popular_items=num_popular_items, user_info=user_info, item_popularity=item_popularity, num_items_per_question=num_items_per_question, save=save, logger=logger)
        self.name = "Greedy"

    def find_next_items(self, num_items=5):
        weights = self.item_popularity.copy()
        for iid in self.seed_item_ids:
            weights[iid] = 0
        choices = np.argsort(weights)[::-1][:num_items]
        return list(choices)
    
    # def recommend_center(self, K=20):
    #     return get_k_closest_items(self.item_embeddings, self.center, k=K)

class DPPSequential(BaseSequential):
    def __init__(self, item_embeddings, max_questions, num_popular_items, user_info, item_popularity, num_items_per_question, gamma=0.6, save=False, logger=None):
        super().__init__(item_embeddings=item_embeddings, max_questions=max_questions, num_popular_items=num_popular_items, user_info=user_info, item_popularity=item_popularity, num_items_per_question=num_items_per_question, save=save, logger=logger)
        self.name = "DPP"
        self.gamma = gamma

    def find_next_items(self, num_items=5):
        N_queried = len(self.seed_item_ids)
        N_unqueried = self.num_popular_items - N_queried
        unqueried_index = [x for x in list(np.arange(self.num_popular_items)) if x not in self.seed_item_ids]
        mapping_to_original_data = self.seed_item_ids + unqueried_index
        weights = np.array(self.item_popularity)
        for iid in self.seed_item_ids:
            weights[iid] = 0
        
        S = self.item_embeddings[mapping_to_original_data] @ self.item_embeddings[mapping_to_original_data].T
        identity_unqueried = np.zeros([self.num_popular_items, self.num_popular_items])
        np.fill_diagonal(identity_unqueried[N_queried : self.num_popular_items, N_queried : self.num_popular_items], 1)
        try:
            term = np.linalg.inv(S + identity_unqueried)[N_queried : self.num_popular_items, N_queried : self.num_popular_items]
        except:
            term = np.linalg.pinv(S + identity_unqueried)[N_queried : self.num_popular_items, N_queried : self.num_popular_items]
        try:
            S_s_t = np.linalg.inv(term) - np.identity(N_unqueried)
        except:
            S_s_t = np.linalg.pinv(term) - np.identity(N_unqueried)

        D = weights[mapping_to_original_data][N_queried:self.num_popular_items] * np.identity(N_unqueried)
        L = self.gamma * S_s_t + (1 - self.gamma) * D
        # L = self.item_embeddings[:self.num_popular_items]@self.item_embeddings[:self.num_popular_items].T
        slt_idx_relative, _, _, _, _, _  = map_inference_dpp_local_search_2(L, k=num_items)
        slt_idx = np.array(mapping_to_original_data[N_queried:self.num_popular_items])[slt_idx_relative]
        # breakpoint()
        return list(slt_idx)


def chebyshev_to_K_dist(chebyshev_center, centers, item, mode, liked=True):
    if mode == 'max' or mode == 'avg' or mode == 'total':
        distance = np.zeros(centers.shape[0])
    else:
        distance = np.full(centers.shape[0], np.inf)
        
    if liked == True:
        diff = item - centers
        diff_norm = np.sum(np.square(centers), axis=1) - np.sum(np.square(item), axis=0)
    else:
        diff = centers - item
        diff_norm = np.sum(np.square(item), axis=0) - np.sum(np.square(centers), axis=1)
    
    d = np.abs(2 * diff @ chebyshev_center + diff_norm) / np.linalg.norm(2 * diff, axis=1)
    # d = np.abs(2 * (centers - item) @ chebyshev_center + np.square(np.linalg.norm(item, 2)) - np.square(np.linalg.norm(centers, 2))) / np.linalg.norm(centers - item, axis=1)

    if mode == 'max':
        distance = np.max(d)
    elif mode == 'min':
        distance = np.min(d)
    elif mode == 'avg':
        distance = np.mean(d)
    elif mode == 'total':
        distance = np.sum(d)
    return distance
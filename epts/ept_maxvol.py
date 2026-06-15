import itertools
import os
import random
from collections import defaultdict, namedtuple
from copy import deepcopy

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.chebysev import *
from src.kmeans import *
from src.methods import *
from src.utils import *
from src.user_interaction import *
from utils import helpers

from .common import _run_calculate_metric, result_holder, calculate_all_metrics


def purge_non_popular_item(info, ec):
    keys = list(info.keys())
    for key in keys:
        val = info[key]
        info[key] = [x for x in val if x < ec.num_samples]
    return info 

def purge_dict(info):
    keys = list(info.keys())
    for key in keys:
        val = info[key]
        if len(val) == 0:
            del info[key]
    
    print(f'after purge, dict has {len(info)} keys left')
    return info 

def run(ec, num_proc, dname, seed, logger):
    logger.info(f"Running dataset: {dname}, method: Maxvol")

    np.random.seed(seed)
    random.seed(seed)

        # data loading and unpacking 
    item_embeddings, item_popularity, user_embeddings = load_data(data=dname, n_samples=ec.num_samples, load_all=False)
    ec.num_samples = min(ec.num_samples, item_embeddings.shape[0])
    item_popularity = normalize_old(item_popularity)
    user_embeddings, item_embeddings = normalize(user_embeddings, item_embeddings)
    rating_matrix = user_embeddings @ item_embeddings.T
    new_user_info = gen_user(user_embeddings[:ec.num_users], item_embeddings, item_popularity, 
                            min_kappa=ec.dataset_params[dname]['kappa'], max_kappa=ec.dataset_params[dname]['kappa']+1,
                            num_favorite_item=ec.num_favs, num_users=ec.num_users, seed=seed,
                            dname=dname)
    train_rating_dict = np.load(f'embeddings/{dname}_train_rating_dict.npy', allow_pickle=True).item()
    category_to_items = np.load(f'embeddings/{dname}_categories_to_items.npy', allow_pickle=True).item()
    
    purge_non_popular_item(train_rating_dict, ec)
    purge_non_popular_item(category_to_items, ec)
    purge_dict(category_to_items)
    

    def _run_maxvol(user_info):
        logger.info("Running method Maxvol...")
        method = RecMaxVolSeedItem(rating_matrix, decompose_rank=5)
        seed_item_ids = method.pick(K=ec.num_items)

        step2 = RandomSequential(item_embeddings, max_questions=0, 
                                num_popular_items=ec.num_popular_items, 
                                num_items_per_question=ec.num_items_per_question, 
                                user_info=user_info, item_popularity=item_popularity.copy(), 
                                logger=logger)
        
        step2.ask_and_update(seed_item_ids)
        num_likes = [len(step2.user_response['liked_list'])]
        num_dislikes = [len(step2.user_response['disliked_list'])]

        pred = step2.recommend_center(K=50)
        gt = step2.gt
        results = [calculate_all_metrics(pred, gt, result_holder)]

        logger.info("Done method Maxvol!")
        method = None
        return results, num_likes, num_dislikes

    jobs_args = []
    for idx in range(ec.num_users):
        # news, is_watcheds, favorite_item_ids _user= new_user_info
        user_info = {}
        user_info['embedding'] = new_user_info['new_users'][idx] 
        user_info['is_watched'] = new_user_info['is_watcheds'][idx]
        user_info['favorite_item_list'] = new_user_info['favorite_item_ids'][idx]

        jobs_args.append((user_info))
    # fix epsilon, varying k

    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(_run_maxvol)(
        jobs_args[i]) for i in range(len(jobs_args)))

    results = [np.zeros(ec.max_rounds+1) for _ in range(len(result_holder._fields))]
    num_likes = np.zeros(ec.max_rounds+1)
    num_dislikes = np.zeros(ec.max_rounds+1)
    for (rets, likes, dislikes) in rets:
        
        num_likes += likes
        num_dislikes += dislikes
        for q_idx, r in enumerate(rets):
            for i, field_name in enumerate(r._fields):
                results[i][q_idx] += getattr(r, field_name)

    temp = {} 
    for i, field_name in enumerate(r._fields):
        temp[field_name] = results[i] / ec.num_users
    temp['num_likes'] = num_likes / ec.num_users
    temp['num_dislikes'] = num_dislikes / ec.num_users

    logger.info(f"Done dataset: {dname}, method: Maxvol!")
    return {"DPE" + f"_seed_{seed}": temp}

def run_ept_maxvol(ec, wdir, datasets, num_proc=4, start_seed=0, end_seed=1, logger=None):

    logger.info("Running ept Maxvol...")
    if datasets is None or len(datasets) == 0:
        datasets = ec.ephase1.all_datasets

    for dname in datasets:
        ret = dict()
        for seed in range(start_seed, end_seed):
            res = run(ec.e_dpe, num_proc, dname, seed, logger)
            for k, v in res.items():
                ret[k] = v
        helpers.pdump(ret, f'{dname}_{start_seed}_{end_seed}.pickle', wdir)

    logger.info("Done ept Maxvol.")

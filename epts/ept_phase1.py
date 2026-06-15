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

from .common import _run_calculate_metric, result_holder


def _run_random(ec, item_embeddings, item_popularity, logger):
    logger.info("Running method random...")
    step1 = RandomSeedItem(ec.num_samples)
    seed_item_ids = step1.pick(K=ec.num_questions_phase1)
    logger.info("Done method random!")
    return seed_item_ids

def _run_greedy(ec, item_embeddings, item_popularity, logger):
    logger.info("Running method greedy...")
    step1 = GreedySeedItem(ec.num_samples, item_popularity[:ec.num_samples].copy())
    seed_item_ids = step1.pick(K=ec.num_questions_phase1)
    logger.info("Done method greedy!")
    return seed_item_ids

def _run_popularity(ec, item_embeddings, item_popularity, logger):
    logger.info("Running method popularity..")
    step1 = BinPopularitySeedItem(ec.num_popular_items, item_popularity.copy())
    seed_item_ids = step1.pick(K=ec.num_questions_phase1)
    logger.info("Done method popularity!")
    return seed_item_ids

def _run_dpp(ec, item_embeddings, item_popularity, logger):
    logger.info("Running method dpp...")
    step1 = DPPSeedItem(item_embeddings, ec.num_popular_items, item_popularity.copy())
    seed_item_ids = step1.pick(K=ec.num_questions_phase1, gamma=ec.method_params['dpp']['gamma'] )
    logger.info("Done method dpp!")
    return seed_item_ids

def _run_kmedoids(ec, item_embeddings, item_popularity, logger):
    logger.info("Running method kmedoids...")
    step1 = KmedoidSeedItem(item_embeddings, ec.method_params['kmedoids']['num_runs'],
                            ec.num_popular_items, item_popularity.copy())
    seed_item_ids = step1.pick(K=ec.num_questions_phase1)
    logger.info("Done method kmedoids!")
    return seed_item_ids


method_map = {
    "random": _run_random,
    "dpp": _run_dpp,
    "greedy": _run_greedy,
    "kmedoids": _run_kmedoids, 
    "popularity": _run_popularity
}


def run(ec, num_proc, dname, mname, seed, logger):
    logger.info(f"Running dataset: {dname}, method: %s...",
                mname)

    np.random.seed(seed)
    random.seed(seed)

        # data loading and unpacking 
    item_embeddings, item_popularity, user_embeddings = load_data(data=dname, n_samples=ec.num_samples, load_all=False)
    ec.num_samples = min(ec.num_samples, item_embeddings.shape[0])
    item_popularity = normalize_old(item_popularity)
    user_embeddings, item_embeddings = normalize(user_embeddings, item_embeddings)
    
    new_user_info = gen_user(user_embeddings[:ec.num_users], item_embeddings, item_popularity, 
                                min_kappa=ec.dataset_params[dname]['kappa'], max_kappa=ec.dataset_params[dname]['kappa']+1,
                                num_favorite_item=ec.num_favs, num_users=ec.num_users, seed=seed,
                                dname=dname)
    
    method = method_map[mname]

    seed_item_ids = method(ec, item_embeddings, item_popularity, logger)

    jobs_args = []
    for idx in range(ec.num_users):
        # news, is_watcheds, favorite_item_ids _user= new_user_info
        user_info = {}
        user_info['embedding'] = new_user_info['new_users'][idx] 
        user_info['is_watched'] = new_user_info['is_watcheds'][idx]
        user_info['favorite_item_list'] = new_user_info['favorite_item_ids'][idx]

        jobs_args.append((idx, seed_item_ids, item_embeddings, item_popularity, user_info, ec, method, logger))

    # fix epsilon, varying k
    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(_run_calculate_metric)(
        *jobs_args[i]) for i in range(len(jobs_args)))

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

    logger.info(f"Done dataset: {dname}, method: {mname}!")
    return {mname + f"_seed_{seed}": temp}


def run_ept_phase1(ec, wdir, datasets, methods,
                    num_proc=4, start_seed=0, end_seed=5, logger=None):

    logger.info("Running ept phase1...")

    if methods is None or len(methods) == 0:
        methods = ec.ephase1.all_methods
    if datasets is None or len(datasets) == 0:
        datasets = ec.ephase1.all_datasets

    for dname in datasets:
        ret = dict()
        for mname in methods:
            for seed in range(start_seed, end_seed):
                res = run(ec.ephase1, num_proc, dname, mname, seed, logger)
                for k, v in res.items():
                    ret[k] = v
        helpers.pdump(ret, f'{dname}_{start_seed}_{end_seed}.pickle', wdir)

    logger.info("Done ept phase1.")



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

from .common import _run_calculate_metric, result_holder, calculate_all_metrics, create_results_holder

def _run_sum(idx, seed_item_ids, item_embeddings, item_popularity, user_info, ec, method, logger):
    logger.info("Evaluating method %s with instance %d...", method, idx)
    step2 = PersonalizedSequential(item_embeddings, max_questions=0, 
                            num_popular_items=ec.num_popular_items, 
                            num_items_per_question=ec.num_items_per_question, 
                            user_info=user_info, item_popularity=item_popularity.copy(), 
                            logger=logger, mode='total')
    step2.ask_and_update(seed_item_ids)
    phase2_item_ids = seed_item_ids.copy()
    num_likes = [len(step2.user_response['liked_list'])]
    num_dislikes = [len(step2.user_response['disliked_list'])]

    pred = step2.recommend_center(K=50)
    gt = step2.gt
    results = [calculate_all_metrics(pred, gt, result_holder)]
    for ro in range(ec.max_rounds):
        item_ids = step2.find_next_items(num_items=ec.num_items_per_question)
        phase2_item_ids += item_ids
        phase2_item_ids = list(set(phase2_item_ids))
        step2.ask_and_update(phase2_item_ids)
        pred = step2.recommend_center(K=50)
        results.append(calculate_all_metrics(pred, gt, result_holder))
        num_likes.append(len(step2.user_response['liked_list']))
        num_dislikes.append(len(step2.user_response['disliked_list']))
    # logger.info("Done evaluate method random with instance %d...", idx)
    return results, num_likes, num_dislikes

def _run_avg(idx, seed_item_ids, item_embeddings, item_popularity, user_info, ec, method, logger):
    logger.info("Evaluating method %s with instance %d...", method, idx)
    step2 = PersonalizedSequential(item_embeddings, max_questions=0, 
                            num_popular_items=ec.num_popular_items, 
                            num_items_per_question=ec.num_items_per_question, 
                            user_info=user_info, item_popularity=item_popularity.copy(), 
                            logger=logger, mode='avg')
    step2.ask_and_update(seed_item_ids)
    phase2_item_ids = seed_item_ids.copy()
    num_likes = [len(step2.user_response['liked_list'])]
    num_dislikes = [len(step2.user_response['disliked_list'])]

    pred = step2.recommend_center(K=50)
    gt = step2.gt
    results = [calculate_all_metrics(pred, gt, result_holder)]
    for ro in range(ec.max_rounds):
        item_ids = step2.find_next_items(num_items=ec.num_items_per_question)
        phase2_item_ids += item_ids
        phase2_item_ids = list(set(phase2_item_ids))
        step2.ask_and_update(phase2_item_ids)
        pred = step2.recommend_center(K=50)
        results.append(calculate_all_metrics(pred, gt, result_holder))
        num_likes.append(len(step2.user_response['liked_list']))
        num_dislikes.append(len(step2.user_response['disliked_list']))
    # logger.info("Done evaluate method random with instance %d...", idx)
    return results, num_likes, num_dislikes


def _run_min(idx, seed_item_ids, item_embeddings, item_popularity, user_info, ec, method, logger):
    logger.info("Evaluating method %s with instance %d...", method, idx)
    step2 = PersonalizedSequential(item_embeddings, max_questions=0, 
                            num_popular_items=ec.num_popular_items, 
                            num_items_per_question=ec.num_items_per_question, 
                            user_info=user_info, item_popularity=item_popularity.copy(), 
                            logger=logger, mode='min')
    step2.ask_and_update(seed_item_ids)
    phase2_item_ids = seed_item_ids.copy()
    num_likes = [len(step2.user_response['liked_list'])]
    num_dislikes = [len(step2.user_response['disliked_list'])]

    pred = step2.recommend_center(K=50)
    gt = step2.gt
    results = [calculate_all_metrics(pred, gt, result_holder)]
    for ro in range(ec.max_rounds):
        item_ids = step2.find_next_items(num_items=ec.num_items_per_question)
        phase2_item_ids += item_ids
        phase2_item_ids = list(set(phase2_item_ids))
        step2.ask_and_update(phase2_item_ids)
        pred = step2.recommend_center(K=50)
        results.append(calculate_all_metrics(pred, gt, result_holder))
        num_likes.append(len(step2.user_response['liked_list']))
        num_dislikes.append(len(step2.user_response['disliked_list']))
    # logger.info("Done evaluate method random with instance %d...", idx)
    return results, num_likes, num_dislikes


def _run_max(idx, seed_item_ids, item_embeddings, item_popularity, user_info, ec, method, logger):
    logger.info("Evaluating method %s with instance %d...", method, idx)
    step2 = PersonalizedSequential(item_embeddings, max_questions=0, 
                            num_popular_items=ec.num_popular_items, 
                            num_items_per_question=ec.num_items_per_question, 
                            user_info=user_info, item_popularity=item_popularity.copy(), 
                            logger=logger, mode='max')
    step2.ask_and_update(seed_item_ids)
    phase2_item_ids = seed_item_ids.copy()
    num_likes = [len(step2.user_response['liked_list'])]
    num_dislikes = [len(step2.user_response['disliked_list'])]

    pred = step2.recommend_center(K=50)
    gt = step2.gt
    results = [calculate_all_metrics(pred, gt, result_holder)]
    for ro in range(ec.max_rounds):
        item_ids = step2.find_next_items(num_items=ec.num_items_per_question)
        phase2_item_ids += item_ids
        phase2_item_ids = list(set(phase2_item_ids))
        step2.ask_and_update(phase2_item_ids)
        pred = step2.recommend_center(K=50)
        results.append(calculate_all_metrics(pred, gt, result_holder))
        num_likes.append(len(step2.user_response['liked_list']))
        num_dislikes.append(len(step2.user_response['disliked_list']))
    # logger.info("Done evaluate method random with instance %d...", idx)
    return results, num_likes, num_dislikes


method_map = {
    "min": _run_min,
    "max": _run_max,
    "avg": _run_avg,
    "sum": _run_sum
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
    new_user_info = gen_user(user_embeddings, item_embeddings, item_popularity, 
                            min_kappa=ec.dataset_params[dname]['kappa'], max_kappa=ec.dataset_params[dname]['kappa']+1, 
                            num_favorite_item=ec.num_favs, num_users=ec.num_users, seed=seed,
                            dname=dname)
    
    method = method_map[mname]
    step1 = DPPSeedItem(item_embeddings, ec.num_popular_items, item_popularity.copy())
    seed_item_ids = step1.pick(K=ec.num_questions_phase1)
    jobs_args = []
    for idx in range(ec.num_users):
        user_info = {}
        user_info['embedding'] = new_user_info['new_users'][idx]
        user_info['is_watched'] = new_user_info['is_watcheds'][idx]
        user_info['favorite_item_list'] = new_user_info['favorite_item_ids'][idx]

        jobs_args.append((idx, seed_item_ids, item_embeddings, item_popularity, user_info, ec, method, logger))

    # fix epsilon, varying k
    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(method)(
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
           


def run_ept_phase2_vary_mode(ec, wdir, datasets, methods,
                    num_proc=4, start_seed=0, end_seed=5, logger=None):

    logger.info("Running ept phase 2 (vary modes)...")
    if methods is None or len(methods) == 0:
        methods = ec.ephase2_vary_mode.all_methods
    if datasets is None or len(datasets) == 0:
        datasets = ec.ephase2_vary_mode.all_datasets 

    for dname in datasets:
        ret = dict()
        for mname in methods:
            for seed in range(start_seed, end_seed):
                res = run(ec.ephase2_vary_mode, num_proc, dname, mname, seed, logger)
                for k, v in res.items():
                    ret[k] = v
        helpers.pdump(ret, f'{dname}_{start_seed}_{end_seed}.pickle', wdir)

    logger.info("Done ept phase 2 (vary modes).")

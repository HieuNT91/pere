from src.utils import * 
from src.methods import *
from src.metrics import *
from collections import defaultdict, namedtuple

def create_results_holder():
    HR = ['HR_{}'.format(i) for i in [1, 3, 5, 7, 10]]
    Recall = ['Recall_{}'.format(i) for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]]
    NDCG = ['NDCG_{}'.format(i) for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]]
    MAP = ['MAP_{}'.format(i) for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]]
    AUC = ['AUC_{}'.format(i) for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]]
    F1 = ['F1_{}'.format(i) for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]]
    MRR = ['MRR_{}'.format(i) for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]]

    field_names = HR + Recall + NDCG + MAP + AUC + F1 + MRR

    return namedtuple('Results', field_names)
 
result_holder = create_results_holder()


def calculate_all_metrics(pred, gt, result_holder):
    default_values = [0 for _ in range(len(result_holder._fields))]
    for i, metric in enumerate(result_holder._fields):
        name, k = metric.split('_')
        k = int(k)
        pred_ = pred[:k]
        default_values[i] = name_to_metrics[name](pred_, gt)
    return result_holder(*default_values)


def _run_calculate_metric(idx, seed_item_ids, item_embeddings, item_popularity, user_info, ec, method, logger):
    logger.info("Evaluating method %s with instance %d...", method, idx)
    step2 = RandomSequential(item_embeddings, max_questions=0, 
                            num_popular_items=ec.num_popular_items, 
                            num_items_per_question=ec.num_items_per_question, 
                            user_info=user_info, item_popularity=item_popularity.copy(), 
                            logger=logger)
    step2.ask_and_update(seed_item_ids)
    phase2_item_ids = seed_item_ids.copy()
    num_likes = [len(step2.user_response['liked_list'])]
    num_dislikes = [len(step2.user_response['disliked_list'])]

    pred = step2.recommend_center(K=50)
    gt = step2.gt
    results = [calculate_all_metrics(pred, gt, result_holder)]
    for ro in range(ec.max_rounds):
        item_ids = step2.find_next_items()
        phase2_item_ids += item_ids
        phase2_item_ids = list(set(phase2_item_ids))
        step2.ask_and_update(phase2_item_ids)
        pred = step2.recommend_center(K=50)
        results.append(calculate_all_metrics(pred, gt, result_holder))
        num_likes.append(len(step2.user_response['liked_list']))
        num_dislikes.append(len(step2.user_response['disliked_list']))

    # logger.info("Done evaluate method random with instance %d...", idx)
    return results, num_likes, num_dislikes

name_to_metrics = {
    'HR': HR,
    'Recall': Recall,
    'MAP': MAP,
    'NDCG': NDCG,
    'AUC': AUC,
    'F1': F1,
    'MRR': MRR
}
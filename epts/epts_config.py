import argparse

from epts.config import Config

class EptPhase1(Config):
    __dictpath__ = 'ec.ephase1'

    all_datasets = ['gowalla', 'yelp2018', 'amazon_books', 'amazon_games_bivae']
    # all_methods = ['random', 'greedy', 'popularity', 'kmedoids', 'dpp']
    all_methods = ['random', 'kmedoids', 'dpp']


    method_params = {
        "dpp": {
            "gamma": 0.3, 

        },
        "kmedoids": {
            "num_runs": 1
        }
    }

    rec_size = 20
    num_favs = 200
    num_questions = 0
    num_samples = 10000
    num_questions_phase1 = 50
    num_items_per_question = 1
    # kappa_range = (17, 21)

    max_rounds = 0


class EptPhase2(Config):
    __dictpath__ = 'ec.ephase2'

    all_datasets = ['gowalla', 'yelp2018', 'amazon_books', 'amazon_games_bivae', 'amazon_books_bivae']
    # all_methods = ['random', 'greedy', 'pere', 'dpp']
    all_methods = ['random', 'dpp', 'pere']


    method_params = {
        "dpp": {
            "gamma": 0.5,
        }
    }

    rec_size = 20
    num_favs = 200

    num_questions = 5
    max_rounds = num_questions
    num_samples = 15000
    num_questions_phase1 = 50
    num_items_per_question = 10
    # kappa_range = (17, 21)


class EptVaryMode(Config):
    __dictpath__ = 'ec.ephase2'

    all_datasets = ['gowalla', 'amazon_books', 'amazon_games_bivae']
    # all_methods = ['random', 'greedy', 'pere', 'dpp']
    all_methods = ['min', 'max', 'avg', 'sum']


    method_params = {
        "dpp": {
            "gamma": 0.5,
        }
    }

    rec_size = 20
    num_favs = 200
    
    num_questions = 5
    max_rounds = num_questions
    num_samples = 15000
    num_questions_phase1 = 50
    num_items_per_question = 10



class EptDPE(Config):
    __dictpath__ = 'ec.dpe'

    all_datasets = ['gowalla', 'yelp2018', 'amazon_books', 'amazon_games_bivae']
    
    rec_size = 20
    num_favs = 200
    
    num_items = 100
    num_samples = 15000
    num_items_per_question = 5
    max_rounds = 0
    


class EptConfig(Config):
    __dictpath__ = 'ec'

    dataset_params = {
        "gowalla": {
            "kappa": 16,
        },
        "yelp2018": {
            "kappa": 20,
        },
        "amazon_books":{
            "kappa": 21
        },
        "amazon_books_bivae":{
            "kappa": 17
        },
        "amazon_games_bivae":{
            "kappa": 17
        }
    }

    ephase1 = EptPhase1()
    ephase2 = EptPhase2()
    ephase2_vary_mode = EptVaryMode()
    e_dpe = EptDPE()

    ephase1.dataset_params = dataset_params
    ephase2.dataset_params = dataset_params
    ephase2_vary_mode.dataset_params = dataset_params
    e_dpe.dataset_params = dataset_params
    ephase1.num_users = ephase2.num_users = ephase2_vary_mode.num_users = e_dpe.num_users = 2
    ephase1.num_popular_items = ephase2.num_popular_items = ephase2_vary_mode.num_popular_items = e_dpe.num_popular_items = 3000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    if args.load is not None:
        EptConfig.from_file(args.load)
    EptConfig.to_file(args.dump, mode=args.mode)
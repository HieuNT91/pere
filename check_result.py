from utils import helpers
import argparse

parser = argparse.ArgumentParser(description='run experiments')

parser.add_argument('--result_file', '-r',  default='amazon_0_1.pickle', type=str)
parser.add_argument('--outdir', '-o', default='results/run_1/ept_phase1', type=str)


args = parser.parse_args()

result_file = args.result_file
outdir = args.outdir

result = helpers.pload(result_file, outdir)

for k, v in result.items():
    print(k)
    for vk, vv in v.items():
        if isinstance(vv, float):
            print(vv)
            #print(vk, ":", vv)
        else:
            #print(vk)
            print(*[round(x, 4) for x in list(vv)])
            #print(vk, ":", *[round(x,4) for x in list(vv)])












# from utils import helpers
# import os 
# import numpy as np

# # all_datasets = ['gowalla', 'yelp2018', 'movielens10m', 'amazon_books']
# # phase1_methods = ['random', 'greedy', 'popularity', 'kmethods', 'dpp']
# # phase1_dir = 'main_result/ept_phase1'
# # all_files = os.listdir(phase1_dir)
# # for dataset in all_datasets:
# #     for fi in all_files:
# #         fi_path = os.path.join(phase1_dir, fi)
# #         if dataset in fi:
# #             result = helpers.pload(fi_path)
# #             runs = list(result.keys())
# #             metrics = list(result[runs[0]].keys())

# #             for phase1_methods 
# #             for run in runs:
                
# #                 avgs = []
# #                 for metric in metrics:
# #                     avgs.append()
# def concatenate_dicts(dicts):
#     # Create an empty dictionary to hold the concatenated dictionaries
#     result_dict = {}
    
#     # Loop over each dictionary in the list of dictionaries
#     for d in dicts:
#         # Update the result dictionary with the key-value pairs of the current dictionary
#         result_dict.update(d)
    
#     return result_dict

# def accumulate_metrics(dicts):
#     # Create an empty dictionary to store the accumulated metrics
#     accumulated = {}
    
#     # Loop over each method_seed in the dictionary
#     for method_seed, metrics in dicts.items():
#         # Parse the method and seed from the method_seed string
#         method, seed = method_seed.split('_seed_')
        
#         # Loop over each metric in the metrics dictionary
#         for metric, value in metrics.items():
#             # Convert the value to a numpy array if it isn't already one
#             if not isinstance(value, np.ndarray):
#                 value = np.array([value])
            
#             # If the metric hasn't been seen before, add it to the accumulated dictionary
#             if metric not in accumulated:
#                 accumulated[metric] = {}
            
#             # If the method hasn't been seen before for this metric, add it to the accumulated dictionary
#             if method not in accumulated[metric]:
#                 accumulated[metric][method] = np.zeros_like(value)
            
#             # Add the value vector or scalar for this seed to the accumulated vector or scalar for this metric and method
#             if value.shape == (1,):
#                 accumulated[metric][method] += value
#             else:
#                 accumulated[metric][method] += value.reshape(-1)
    
#     # Compute the average vector or scalar for each metric and method
#     for metric, methods in accumulated.items():
#         for method, values in methods.items():
#             if values.shape == (1,):
#                 methods[method] /= len(dicts)
#             else:
#                 methods[method] /= len(dicts)
#                 methods[method] = methods[method].reshape(-1)
    
#     return accumulated


# all_datasets = ['gowalla', 'yelp2018', 'movielens10m', 'amazon_books']
# phase1_methods = ['random', 'greedy', 'popularity', 'kmethods', 'dpp']
# phase1_dir = 'main_result/ept_phase1'
# phase2_dir = 'main_result/ept_phase2'
# phasetune_dir = 'main_result/1_item_per_question'
# all_files = os.listdir(phase1_dir)


# for fi in all_files:
#     print('='*100)
#     fi_path = os.path.join(phase1_dir, fi)
#     dicts = helpers.pload(fi_path)
#     # breakpoint()
#     averaged_metrics = accumulate_metrics(dicts)
#     for metric, methods in averaged_metrics.items():
#         print()
#         print(f"Metric: {metric}")
#         for method, values in methods.items():
#             print(f"Method {method}: {values}")

# # # Define a list of dictionaries with the structure {'methodA_seed': {'metricA': float, 'metricB': float}, 'methodB_seed': {'metricA': float, 'metricB': float}}
# # dicts = [{'methodA_1': {'metricA': 0.5, 'metricB': 0.2}, 'methodB_1': {'metricA': 0.3, 'metricB': 0.7}}, 
# #          {'methodA_2': {'metricA': 0.4, 'metricB': 0.8}, 'methodB_2': {'metricA': 0.2, 'metricB': 0.5}}, 
# #          {'methodA_3': {'metricA': 0.6, 'metricB': 0.3}, 'methodB_3': {'metricA': 0.4, 'metricB': 0.6}}]

# # Call the accumulate_metrics function to compute the average metrics across all seeds and methods



# # all_files = os.listdir(phase2_dir)

# # dicts = []
# # for fi in all_files:
# #     fi_path = os.path.join(phase2_dir, fi)
# #     dicts.append(helpers.pload(fi_path))
# #     # breakpoint()

# # averaged_metrics = accumulate_metrics(dicts[5])
# # for metric, methods in averaged_metrics.items():
# #     print()
# #     print(f"Metric: {metric}")
# #     for method, values in methods.items():
# #         print(f"Method {method}: {values}")

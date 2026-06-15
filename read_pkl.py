from utils import helpers
import argparse
import csv 
import numpy as np
parser = argparse.ArgumentParser(description='run experiments')

parser.add_argument('--result-file', '-r',  default='amazon_0_1.pickle', type=str)
parser.add_argument('--outdir', '-o', default='results/run_1/ept_phase1', type=str)
parser.add_argument('--phase2', action='store_true', default=False)
parser.add_argument('--maxvol', action='store_true', default=False)
parser.add_argument('--seed-list', dest='seeds', nargs='*')
args = parser.parse_args()

result_file = args.result_file
outdir = args.outdir
dname = "_".join(args.result_file.split("_")[:3])
result = helpers.pload(result_file, outdir)
# print(result)
mnames_seeds = [x.split("_seed_") for x in list(result.keys())]
mnames = list(set([x[0] for x in mnames_seeds]))

if 'popularity' in mnames:
    mnames.remove('popularity')
seeds = list(set([x[1] for x in mnames_seeds]))
metric_list = list(result[f'{mnames[0]}_seed_{seeds[0]}'].keys())
# metric_list = ['HR_3', 'HR_5', 'Recall_5', 'Recall_10', 'NDCG_5', 'NDCG_10', 'MAP_5', 'MRR_30']
max_rounds = len(result[f'{mnames[0]}_seed_{seeds[0]}'][metric_list[0]])
# breakpoint()
# for mname_seed, v in result.items():
#     print()
#     print(mname_seed)
#     print([(name, val) for name, val in v.items()])
 
best_metric = []
if not args.phase2:
    for metric in metric_list:
        avgs = []
        stds = []
        for mname in mnames:
            metric_vals = []
            for seed in seeds:
                key = f'{mname}_seed_{seed}'
                metric_vals.append(float(result[key][metric][0]))
            avgs.append(np.average(metric_vals))
            stds.append(np.std(metric_vals))
        
        best_idx = np.argmax(avgs)
        best_mname = mnames[best_idx]

        for i, mname in enumerate(mnames):
            key = f'{mname}_avg'
            if key not in result:
                result[key] = {}
            if mname != best_mname:
                result[key][metric] = '{:.4f} $\pm$ {:.3f}'.format(avgs[i], stds[i])
            else:
                result[key][metric] = '\\textbf{{{:.4f} $\pm$ {:.3f}}}'.format(avgs[i], stds[i])
        if best_mname == 'dpp':
            best_metric.append(metric)
else:
    for metric in metric_list:
        for q_idx in range(max_rounds):
            avgs = []
            stds = []
            for mname in mnames:
                metric_vals = []
                for seed in seeds:
                    key = f'{mname}_seed_{seed}'
                    metric_vals.append(float(result[key][metric][q_idx]))
                avgs.append(np.average(metric_vals))
                stds.append(np.std(metric_vals))
            
            best_idx = np.argmax(avgs)
            best_mname = mnames[best_idx]

            for i, mname in enumerate(mnames):
                key = f'{mname}_avg'
                if key not in result:
                    result[key] = {}
                if metric not in result[key]:
                    result[key][metric] = np.zeros(max_rounds, dtype=object)
                if mname != best_mname:
                    result[key][metric][q_idx] = '{:.4f} $\pm$ {:.3f}'.format(avgs[i], stds[i])
                else:
                    result[key][metric][q_idx] = '\\textbf{{{:.4f} $\pm$ {:.3f}}}'.format(avgs[i], stds[i])
            
            if best_mname == 'pere' and q_idx == 10:
                best_metric.append(metric)

print(dname, ":", *best_metric)

metric_list = [x.replace('MAP_5', 'MAP') for x in metric_list]
metric_list = [x.replace('MRR_30', 'MRR') for x in metric_list]
metric_list = [x.replace('_', '@') for x in metric_list]
# if 'pere' in mnames:
#     mnames = ['pere', 'random', 'dpp']
# else:
#     mnames = ['random', 'kmedoids', 'dpp']

if not args.phase2:
    with open(f'Phase1_{dname}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Method'] + metric_list
        writer.writerow(header)
        for mname in mnames:
            row = [f'{mname}']
            for metric_name, metric_vals in result[f'{mname}_avg'].items():
                row.append(metric_vals)
            writer.writerow(row)
    print('result saved to', f'Phase1_{dname}.csv')
else:
    mnames_str = '_'.join(mnames)
    if args.maxvol:
        mnames_str = mnames_str.replace('DPE', 'maxvol')
    with open(f'Phase2_{dname}_{mnames_str}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Method'] + metric_list
        writer.writerow(header)
        for q_idx in range(max_rounds):
            # if q_idx != 5:
            #     continue
            for mname in mnames:
                row = [f'{mname}_avg_'+ str(q_idx)]
                print(row)
                for metric_name, metric_vals in result[f'{mname}_avg'].items():
                    row.append(metric_vals[q_idx])
                writer.writerow(row)
    print('result saved to', f'Phase2_{dname}_{mnames_str}.csv')
# def save_metrics_to_csv(metrics, methods, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
        
#         # Write the header row
#         header = ['Method'] + metrics
#         writer.writerow(header)
        
#         # Write each method and its corresponding metrics
#         for method in methods:
#             row = [method] + metrics
#             writer.writerow(row)

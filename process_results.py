"""
Created on Tue Nov 17 2020
Last updated on Sun 27 2020
@author: jgajardo
Code structure and snippets borrowed from Patrick Kridger.
Description: Parse results from output files of experiments runs to plots or summary tables per metric.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pathlib
import os
import ast
import argparse
import statistics
import itertools
import json

here = pathlib.Path(__file__).resolve().parent

def read_run(filename):
    ''' Process the data from the ouput text or json files and returns metrics in lists across epochs'''
    metadata = []
    config = []
    contents = []
    run_results = {'epochs': [],
                'epoch times': [],
                'nfes': [],
                'bnfes': [],
                'total nfes': [],
                'train loss': [],
                'train acc': [],
                'train f1': [],
                'train perclass f1': [],
                'train confusion': [],
                'val loss': [],
                'val acc': [],
                'val f1': [],
                'val perclass f1': [],
                'val confusion': [],
                'test acc': [],
                'test f1': [],
                'best epoch': [],
                'train time': []}

    if filename.suffix == '.json':
        with open(filename, 'r') as f:
            content = json.load(f)
            config = content['config']
            for epoch_results in content['history']:
                run_results['epochs'].append(epoch_results['epoch'])
                run_results['epoch times'].append(epoch_results['epoch_time'])
                run_results['nfes'].append(epoch_results['nfes'])
                run_results['bnfes'].append(epoch_results['bnfes'])
                run_results['total nfes'].append(epoch_results['total_nfes'])
                run_results['train loss'].append(epoch_results['train_metrics']['loss'])
                run_results['train acc'].append(epoch_results['train_metrics']['accuracy'])
                run_results['train f1'].append(epoch_results['train_metrics']['f1_score'])
                run_results['train perclass f1'].append(epoch_results['train_metrics']['perclass_f1'])
                run_results['train confusion'].append(epoch_results['train_metrics']['confusion'])
                run_results['val loss'].append(epoch_results['val_metrics']['loss'])
                run_results['val acc'].append(epoch_results['val_metrics']['accuracy'])
                run_results['val f1'].append(epoch_results['val_metrics']['f1_score'])   
                run_results['val perclass f1'].append(epoch_results['val_metrics']['perclass_f1'])
                run_results['val confusion'].append(epoch_results['val_metrics']['confusion'])

            # Test metrics
            run_results['test acc'].append(content['test_metrics']['test_accuracy'])
            run_results['test f1'].append(content['test_metrics']['test_f1_score'])
            run_results['best epoch'].append(content['test_metrics']['best_epoch'])
            run_results['train time'].append(content['test_metrics']['train_time'])
            
    elif filename.suffix == '.txt': # for backward compatibility when my output was in txt files and didn't have confusion or perclass F1-scores on it (so could read per line)
        with open(filename, 'r') as f: 
            lines = f.readlines()
            for line in lines:
                if not line.startswith('{'):
                    metadata.append(line)
                    continue
                contents.append(ast.literal_eval(line))

        for entry in contents:
            if 'config' in entry:
                config = entry['config']
                continue
            if 'epoch' in entry:
                run_results['epochs'].append(entry['epoch'])
                run_results['epoch times'].append(entry['epoch_time'])
                run_results['nfes'].append(entry['nfes'])
                run_results['bnfes'].append(entry['bnfes'])
                run_results['total nfes'].append(entry['total_nfes'])
                run_results['train loss'].append(entry['train_metrics']['loss'])
                run_results['train acc'].append(entry['train_metrics']['accuracy'])
                run_results['train f1'].append(entry['train_metrics']['f1_score'])
                run_results['val loss'].append(entry['val_metrics']['loss'])
                run_results['val acc'].append(entry['val_metrics']['accuracy'])
                run_results['val f1'].append(entry['val_metrics']['f1_score'])
            if 'test_accuracy' in entry:
                run_results['test acc'].append(entry['test_accuracy'])
                run_results['test f1'].append(entry['test_f1_score'])
                run_results['train time'].append(entry['train_time'])

    return run_results, config, metadata

def read_folder(metrics, folder_name='', mode='value'):
    '''mode: options are value (scalar) or history (series of values)'''
    loc = here / folder_name
    if not isinstance(metrics, list):
        metrics = [metrics]
    folder_results = {metric: [] for metric in metrics}
    folder_results['models'] = []
    folder_results['params'] = []
    if mode == 'history':
        folder_results['epochs'] = []
    for filename in os.listdir(loc):
        run_results, config, _ = read_run(loc/filename)
        folder_results['models'].append(config['model'])
        folder_results['params'].append(config['num_parameters'])
        if mode == 'history':
            folder_results['epochs'].append(run_results['epochs'])
        for metric in metrics:
            folder_results[metric].append(run_results[metric])
   
    return folder_results

def plot_history(metric, folder_name=''):
    ''' Plots one metric history for each model for all files in a folder. Can't handle subfolders.'''
    folder_results = read_folder(metric, folder_name, mode='history')
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # work around OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    # Reorganize results by model and store series to plot
    models = set(folder_results['models'])
    models_results = {model: [] for model in models}
    nruns = len(next(iter(folder_results.values())))
    for run in range(nruns):
        model = folder_results['models'][run]
        times = folder_results['epochs'][run]
        values = folder_results[metric][run]
        models_results[model].append((times, values))
    # Make plot for each model
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # max 7 models
    for color, model in zip(colors, models):
        results = models_results[model]
        all_times = set() # unique times
        for times, _ in results:
            all_times.update(times)
        all_times = sorted(list(all_times))    
        all_values = [[] for _ in range(len(all_times))]
        for times, values in results:
            # some runs may have finished earlier than others
            assert times == all_times[:len(times)]
            for i, value in enumerate(values):
                all_values[i].append(value)
        means = [statistics.mean(entry) for entry in all_values]
        stds = [statistics.stdev(entry) if len(entry) > 1 else 0 for entry in all_values] # uses bessel's correction
        plt.plot(all_times, means, label=model, color=color)
        plt.fill_between(all_times,
                         [mean + 0.2 * std for mean, std in zip(means, stds)],
                         [mean - 0.2 * std for mean, std in zip(means, stds)],
                         color=color,
                         alpha=0.5)
    plt.title(f'{metric.capitalize()} during training')
    plt.xlabel('Epochs')
    plt.ylabel('Epoch duration (s)' if metric == 'epoch times' else f'{metric.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def table(metrics, folder_name=''):
    "metrics: list of metrics to print. Can handle subfolders."
    for folder in os.listdir(folder_name):
        folder_path = os.path.join(folder_name, folder)
        print('Folder:', folder_path)
        folder_results = read_folder(metrics, folder_path)
        # Reorganize results by model
        models = set(folder_results['models'])
        models_results = {model: {metric: [] for metric in metrics} for model in models}
        param_results = {model: {'params': []} for model in models}
        nruns = len(next(iter(folder_results.values())))
        for run in range(nruns):
            model = folder_results['models'][run]
            param_results[model]['params'].append(folder_results['params'][run])
            for metric in metrics:
                models_results[model][metric].append(folder_results[metric][run])

        # Print summary table for each metric and model
        min_length = min([len(next(iter(models_results[model].values()))) for model in models_results]) # nruns for model with least of them
        print('Num samples:', min_length)
        for metric in metrics:
            print(f'\n{metric.capitalize()}' + ' (s):' if metric == 'train time' else f'\n{metric.capitalize()}:')
            sorted_results = []
            for model, content in models_results.items():
                # pad metrics of shorter runs
                content[metric] = np.array(list(itertools.zip_longest(*content[metric], fillvalue=0))).T
                max_length = max([len(run) for run in content[metric]]) # longest run
                assert max_length == content[metric].shape[1]
                if 'nfes' in metric:
                    sorted_results.append((model, torch.Tensor(content[metric]).sum(dim=1))) # sum nfes over epochs. Convert from numpy ndarray to tensor because of torch.tensor.std uses bessel correction.
                elif metric == 'val acc' or metric == 'val f1':
                    if 'val f1' in metrics:
                        max_value, idx_best = torch.Tensor(content['val f1']).max(dim=1)
                    else:
                        max_value, idx_best  = torch.Tensor(content[metric]).max(dim=1)        
                    sorted_results.append((model, torch.Tensor(content[metric]).gather(dim=1, index=idx_best.unsqueeze(-1))))
                else:
                    sorted_results.append((model, torch.Tensor(content[metric])))
            sorted_results.sort(key=lambda x: -x[1].mean())
            for model, values in sorted_results:
                values = values[:min_length]
                params = np.mean(param_results[model]['params'], dtype=int) # average should be redundant
                print(f'{model.upper() :6}: mean: {values.mean():.3f} std: {values.std():.3f} min: {values.min():.3f} max: {values.max():.3f}| params: {params}')
        print('\n')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default=["test acc", "test f1", "val f1", "val acc"], nargs='+', choices=["test acc", "test f1", "train time", "nfes", "bnfes", "total nfes", "val f1", "val acc"], help='List of metrics to compute in a table. If val acc is passed, val f1 must be passed before and results at epoch with highest val f1 are retrieved for each run. If plot option only the first metric is considered. [default=%(default)s].' )
    parser.add_argument('--plot-metric', type=str, default="train loss", choices=["epoch times", "nfes", "bnfes", "total nfes", "train loss", "train acc", "train f1", "val loss", "val acc", "val f1"], help='Metric to plot [default=%(default)s].' )
    parser.add_argument('--form', type=str, default='table', choices=['table', 'plot'], help='Show results as a table or a plot [default=%(default)s].')
    parser.add_argument('--folder', type=str, default='results', help='Folder where to read the experiments results from (all models results can be there) [default=current directory].')
    args = parser.parse_args()

    if args.form == 'table':
        table(args.metrics, args.folder)
    else:
        plot_history(args.plot_metric, args.folder)
"""
Created on Fri Oct 23 2020
@author: jgajardo
Code structure and snippets borrowed from Patrick Kridger and Nando Metzger.
Versions:
v1: 04.11.2020. Added F1 score, evaluate metrics functions, clearer output.
v2: 06.11.2020. Added early stopping, modified architecture, L2 regularization, named experiments output, modified get data function to include clouds mask, added some helper functions.
v3: 09.11.2020. Added argparse, modified early stopping conditions.
v4: 11.09.2020. LR as an argument and LR decay, added wandb logger. 16.11.2020 lr decay as an option and improved save text file name format, added samples option. 18.11.2020 Added num workers. Changed eps to 10^-10. 20.11.2020. Added print missing values rate if clouds mask options in data.
v5: 20.11.2020. Added experiment ID and save and load checkpoints. Added memory pinning. Using noskip datasets and return mask too in get_data function. Log gradients.
v6: 06.12.2020. Modified data preprocessing and interpolation, added option for reduced dataset, times to use, interpolation method and save coeffs as dataset.
v7: 09.12.2020. Added batch norm and root data path argument. Added tqdm progress bar for measuring epochs speed. Added faster dataloader option.
v8: 14.12.2020. Added gradient clipping and replaced batch norm by layer norm options. 15.12.2020. Added RNN baseline models and semilog speed up option. 23.12. Added odernn baseline, atol/rtol options and nfes + time logging. 24.12. Changed results save name and argument. 29.12. Added activation functions, minor changes to defaults for lr decay and grid search option. 
v9: 05.01.2021. Change in test metrics best model and to regularization argument. 11.01. Added confusion matrix and per class F1 scores in evaluate metrics function. Changes at test time. 13.01. Output as dictionary and save to json file. 15.01. Changes in data preprocessing, added observational mask as option (intensity argument). Added only CELU as option for activation functions, fixed layer norm in CDEFunc, some pep8 conformity. 21.01. Log batch training loss and batch iteration counter. 24.01. Changes in data preprocessing (one-channel obs mask, normalize new channels).
v10: 27.01.2021. Included option for squared exponential kernel intepolation. 30.01. Modified data preprocessing: added padding the data for ncde model and drop last timestep. 08.02. Alternative loss function, small modifications in cdeint_options and seminorm to work with torchcde 0.2.0 and added ncde_stacked model.

"""

###  Import libraries
# Standard 
import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sn

# Data
import h5py
import json

# General utility
import warnings
import argparse
import itertools
import copy
import time
import random
import math
import pdb
import tqdm

# System
import os
import sys
import glob
import pathlib

# ML
import sklearn.metrics
import torch
import torchdiffeq # >= 0.2.1
import torchcde # >= 0.2.0
import wandb

# Own modules
import quadratic_kernel_interpolation

# Check path
here = pathlib.Path(__file__).resolve().parent

# Ignore warnings
warnings.filterwarnings("ignore")


################################################################################################
# Swiscrop labels
swisscrop_labels_names = ['No Label','Maize', 'Meadow', 'Pasture', 'Potatoes', 'Spelt', 'Sugarbeets', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat', 'Winter barley', 'Winter rapeseed', 'Winter wheat']
swisscrop_labels = [0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]

# TUM labels
TUM_labels_names = [ "other", "corn", "meadow", "asparagus", "rape", "hop", "summer oats", "winter spelt", "fallow", "winter wheat",
                    "winter barley", "winter rye", "beans", "winter triticale", "summer barley", "peas", "potatoe", "soybeans", "sugar beets" ]
TUM_labels_dict = {k: i for i, k in enumerate(TUM_labels_names)}
reverse_TUM_labels_dict = {v: k for k, v in TUM_labels_dict.items()}


# Get processed data from TUM dataset (TODO add script with Nando's code of data processing!)
def get_data(absolute_data_directory_path, use_noskip=False, reduced=False, ntrain=None, nval=None, use_model='ncde', intensity=False):
    # Read dataset
    noskip = 'noskip' if use_noskip else ''
    
    times_dataset = h5py.File(os.path.join(absolute_data_directory_path, 'time.hdf5'), 'r')
    train_dataset = h5py.File(os.path.join(absolute_data_directory_path, f'train{noskip}.hdf5'), 'r')
    val_dataset = h5py.File(os.path.join(absolute_data_directory_path, f'eval{noskip}.hdf5'), 'r')
    test_dataset = h5py.File(os.path.join(absolute_data_directory_path, f'test{noskip}.hdf5'), 'r')

    # Extract data, labels and mask
    data = {} # dictionary for storing the data, mask and labels)
    data['times'] = torch.Tensor(times_dataset['tt'][:])

    data['train_data'] = torch.Tensor(train_dataset['data'][:])
    data['val_data'] = torch.Tensor(val_dataset['data'][:])
    data['test_data'] = torch.Tensor(test_dataset['data'][:])

    data['train_labels'] = torch.Tensor(train_dataset['labels'][:])
    data['val_labels'] = torch.Tensor(val_dataset['labels'][:])
    data['test_labels'] = torch.Tensor(test_dataset['labels'][:])

    data['train_mask'] = torch.Tensor(train_dataset['mask'][:])
    data['val_mask'] = torch.Tensor(val_dataset['mask'][:])
    data['test_mask'] = torch.Tensor(test_dataset['mask'][:])

    # Return only sample of data if asked for
    data['train_data'] = data['train_data'][:ntrain]
    data['train_labels'] = data['train_labels'][:ntrain]
    data['train_mask'] = data['train_mask'][:ntrain]
    data['val_data'] = data['val_data'][:nval]
    data['val_labels'] = data['val_labels'][:nval]
    data['val_mask'] = data['val_mask'][:nval]
    data['test_data'] = data['test_data'][:nval]
    data['test_labels'] = data['test_labels'][:nval]
    data['test_mask'] = data['test_mask'][:nval]

    # Reduce data features if required (only keep features of central pixel of the 3x3 neighbourhood)
    if reduced:
        data['train_data'] = data['train_data'][:, :, 4:-1:9]
        data['val_data'] = data['val_data'][:, :, 4:-1:9]
        data['test_data'] = data['test_data'][:, :, 4:-1:9]

        data['train_mask'] = data['train_mask'][:, :, 4:-1:9]
        data['val_mask'] = data['val_mask'][:, :, 4:-1:9]
        data['test_mask'] = data['test_mask'][:, :, 4:-1:9]

    # Processing
    if use_model == 'ncde' or use_model == 'ncde_stacked': # TODO: just remove this if-else clause if later I decide to do "same imputation method for all models" (interpolation by torchcde)
        
        train_mask = data['train_mask'].to(bool)
        val_mask = data['val_mask'].to(bool)
        test_mask = data['test_mask'].to(bool)
        
        # Impute NaNs in non-observed pixels (0=False=unobserved due to bad weather)
        data['train_data'][train_mask == False] = float('nan')
        data['val_data'][val_mask == False] = float('nan')
        data['test_data'][test_mask == False] = float('nan')

        for i in ['train_data', 'val_data', 'test_data']:
            # Concatenate observational mask of features as extra features
            if intensity: 
                # Check that observational mask has all channels identical 
                obs_mask = (~torch.isnan(data[i])).cumsum(dim=-2) # cumsum in time dimension (cumulative mask, when doing dX/dt just the simple observational mask will be recovered)
                obs_mask = obs_mask[:, :, 0].unsqueeze(-1).to(dtype=torch.float32) # pick any channel they are all the same
                
                # Normalize obs_mask and append as a channel
                obs_mask = (obs_mask - obs_mask.mean()) / (obs_mask.std() + 1e-6)
                data[i] = torch.cat([data[i], obs_mask], dim=-1) # concatenate in features dimension
    
            # Concatenate time as the first feature
            t = data['times'].unsqueeze(0).repeat(data[i].size(0), 1).unsqueeze(-1)
            t = (t - t.mean()) / (t.std() + 1e-6) # better if it's 0-centered and scaled by its std like the rest of the data
            data[i] = torch.cat([t, data[i]], dim=-1)

            # Pad backwards and forward first and last valid timesteps
            if intensity:
                num_non_intensity_channels = data[i].size(-1) - 1 # obs_mask is just one channel
            else:
                num_non_intensity_channels = data[i].size(-1)
            
            data[i] = fill_first_nonnan_backward(data[i], num_non_intensity_channels)
            data[i] = fill_last_nonnan_forward(data[i], num_non_intensity_channels)

    else: # baselines

        for i in ['train_data', 'val_data', 'test_data']:

            # Concatenate observational mask of features as extra features
            if intensity:
                # Check that observational mask has all channels identical 
                obs_mask = (data[i] != 0.0)
                obs_mask = obs_mask[:, :, 0].unsqueeze(-1).to(dtype=torch.float32) # pick any channel they are all the same

                # Normalize obs_mask and append as a channel
                obs_mask = (obs_mask - obs_mask.mean()) / (obs_mask.std() + 1e-6)
                data[i] = torch.cat([data[i], obs_mask], dim=-1) # concatenate in features dimension
            
            # Concatenate time as the first feature
            t = data['times'].unsqueeze(0).repeat(data[i].size(0), 1).unsqueeze(-1)
            t = (t - t.mean()) / (t.std() + 1e-6) # better if it's 0-centered and scaled by its std like the rest of the data
            data[i] = torch.cat([t, data[i]], dim=-1)
    
    # Drop last and first 4 timesteps because the only have invalid observations
    data['train_data'] = data['train_data'][:, :-1, :]
    data['val_data'] = data['val_data'][:, :-1, :]
    data['test_data'] = data['test_data'][:, :-1, :]
    data['times'] = data['times'][:-1]       

    # Print missing values rate
    train_missing_rate = get_missing_values_rate(data['train_data'])
    val_missing_rate = get_missing_values_rate(data['val_data'])
    test_missing_rate = get_missing_values_rate(data['test_data'])
    print(f'Train data has {train_missing_rate * 100 :0.3f}% of missing values.')
    print(f'Val data has {val_missing_rate * 100 :0.3f}% of missing values.')
    print(f'Test data has {test_missing_rate * 100 :0.3f}% of missing values.')

    return data


def fill_last_nonnan_forward(x, num_non_intensity_channels): # warning: it only works for tensors of ndim=3 and to fill in the dim=1.
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    tmp = np.flip(x, axis=1)
    idxs = np.argmax(~np.isnan(tmp[..., :num_non_intensity_channels]), axis=1)
    idx = np.min(idxs[:, 1:], axis=1)
    for i, sample in enumerate(tmp):
        tmp[i, :idx[i], :num_non_intensity_channels] = np.tile(sample[idx[i], :num_non_intensity_channels], (idx[i], 1))
    x_mod = np.flip(tmp, axis=1)
    x_mod = torch.from_numpy(x)
    assert isinstance(x_mod, torch.Tensor)
    return x_mod


def fill_first_nonnan_backward(x, num_non_intensity_channels): # warning: it only works for tensors of ndim=3 and to fill in the dim=1. Also inefficient because of for loop.
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    idxs = np.argmax(~np.isnan(x[..., :num_non_intensity_channels]), axis=1)
    idx = np.min(idxs[:, 1:], axis=1)
    for i, sample in enumerate(x):
        x[i, :idx[i], :num_non_intensity_channels] = np.tile(sample[idx[i], :num_non_intensity_channels], (idx[i], 1))
    x_mod = torch.from_numpy(x)
    assert isinstance(x_mod, torch.Tensor)
    return x_mod


def get_missing_values_rate(data_tensor):
    ''' Helper function for computing the missing values rate.'''
    assert isinstance(data_tensor, torch.Tensor)
    missing_rate = data_tensor[torch.isnan(data_tensor)].numel()/data_tensor.numel()
    return missing_rate


def get_interpolation_coeffs(directory, data, times, use_noskip, reduced, interpolation_method='cubic'):
    # Create new folder for storing coefficients as datasets (interpolation is expensive)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Dataset name
    noskip = 'noskip' if use_noskip else ''
    red = 'red' if reduced else ''
    timing = 'eqspaced' if times is None or interpolation_method == 'rectilinear' else 'irrspaced' # quick naming fix. TODO: if another equally spaced time series timestamps is in times it wouldn't name it properly...
    coeffs_filename = f'{interpolation_method}_coeffs{noskip}{red}_{timing}.hdf5'
    absolute_coeffs_filename_path = os.path.join(directory, coeffs_filename)
    
    # Interpolate and save, or load it for use if it exists
    coefficients = {}
    if not os.path.exists(absolute_coeffs_filename_path):
        print(f'Interpolating by {interpolation_method} method...')
        if interpolation_method == 'cubic':
            coefficients['train_coeffs'] = torchcde.natural_cubic_coeffs(data['train_data'], t=times)
            coefficients['val_coeffs'] = torchcde.natural_cubic_coeffs(data['val_data'], t=times)
            coefficients['test_coeffs'] = torchcde.natural_cubic_coeffs(data['test_data'], t=times)
        
        elif interpolation_method == 'linear':
            coefficients['train_coeffs'] = torchcde.linear_interpolation_coeffs(data['train_data'], t=times)
            coefficients['val_coeffs'] = torchcde.linear_interpolation_coeffs(data['val_data'], t=times)
            coefficients['test_coeffs'] = torchcde.linear_interpolation_coeffs(data['test_data'], t=times)
        
        elif interpolation_method == 'rectilinear': # rectifilinear doesn't work when passing time argument or when time channel has NaNs
            if timing == 'irrspaced': print('Warning: will do default equally spaced time array instead, rectifilinear interpolation currently works with it only.')
            coefficients['train_coeffs'] = torchcde.linear_interpolation_coeffs(data['train_data'], rectilinear=0)
            coefficients['val_coeffs'] = torchcde.linear_interpolation_coeffs(data['val_data'], rectilinear=0)
            coefficients['test_coeffs'] = torchcde.linear_interpolation_coeffs(data['test_data'], rectilinear=0)

        elif interpolation_method == 'SEkernel':
            if timing == 'irrspaced':
                coefficients['train_coeffs'] = quadratic_kernel_interpolation.quadratic_kernel_interpolation_coeffs(x=data['train_data'], t=times)
                coefficients['val_coeffs'] = quadratic_kernel_interpolation.quadratic_kernel_interpolation_coeffs(x=data['val_data'], t=times)
                coefficients['test_coeffs'] = quadratic_kernel_interpolation.quadratic_kernel_interpolation_coeffs(x=data['test_data'], t=times)
            else:
                raise NotImplementedError('Original timestamps must be used with this interpolation method, please make sure that the time_default command line argument is set to False.')
    
        # Save coefficients in the new directory
        print('Saving interpolation coefficients ...')
        train_nobs, train_ntimes, train_nfeatures = coefficients['train_coeffs'].shape
        val_nobs, val_ntimes, val_nfeatures = coefficients['val_coeffs'].shape
        test_nobs, test_ntimes, test_nfeatures = coefficients['test_coeffs'].shape

        hdf5_coeffs = h5py.File(absolute_coeffs_filename_path , mode='w')
        hdf5_coeffs.create_dataset('train', (train_nobs, train_ntimes, train_nfeatures), np.float, data=coefficients['train_coeffs'])
        hdf5_coeffs.create_dataset('val', (val_nobs, val_ntimes, val_nfeatures), np.float, data=coefficients['val_coeffs'])
        hdf5_coeffs.create_dataset('test', (test_nobs, test_ntimes, test_nfeatures), np.float, data=coefficients['test_coeffs'])
    
    else:
        print(f'Loading interpolation coefficients from: {absolute_coeffs_filename_path}')
        coeffs_dataset = h5py.File(absolute_coeffs_filename_path, mode='r')
        coefficients['train_coeffs'] = torch.Tensor(coeffs_dataset['train'][:])
        coefficients['val_coeffs'] = torch.Tensor(coeffs_dataset['val'][:])
        coefficients['test_coeffs'] = torch.Tensor(coeffs_dataset['test'][:])

    train_coeffs = coefficients['train_coeffs']
    val_coeffs = coefficients['val_coeffs'] 
    test_coeffs = coefficients['test_coeffs'] 
    print(f'Train data interpolation coefficients shape: {train_coeffs.shape}')
    print(f'Validation data interpolation coefficients shape: {val_coeffs.shape}')
    print(f'Test data interpolation coefficients shape: {test_coeffs.shape}')
    return coefficients


def build_data_path(coeffs, times, interpolation_method):
    if interpolation_method == 'cubic':
        X = torchcde.NaturalCubicSpline(coeffs, t=times)
        cdeint_options = {}

    elif interpolation_method == 'linear':
        X = torchcde.LinearInterpolation(coeffs, t=times)
        cdeint_options = dict(jump_t=X.grid_points)

    elif interpolation_method == 'rectilinear': # rectifilinear doesn't work when passing time argument
        X = torchcde.LinearInterpolation(coeffs)
        cdeint_options = dict(jump_t=X.grid_points)

    elif interpolation_method == 'SEkernel':
        X = quadratic_kernel_interpolation.QuadraticKernelInterpolation(coeffs=coeffs, t=times)
        cdeint_options = {}

    return X, cdeint_options


def plot_interpolation_path(coefficients, dataset, times, interpolation_method, n=None):
    print('Plotting interpolation of a sample for sanity check...')
    coeffs = coefficients[f'{dataset}_coeffs']
    if n is None or n > coeffs.size(0):
        n = np.random.randint(0, coeffs.size(0))
    coeffs = coeffs[n].unsqueeze(0)

    X = build_data_path(coeffs, times, interpolation_method)[0]
    t = torch.linspace(0., X.interval[-1], 1001) # plot a lot of points to see the true shape of interpolation (discontinuities in dX for linear and the smoothness of cubic) 

    print('t for plot:', t, t.shape)
    x = X.evaluate(t).squeeze(0)
    print('sample x for plot:', x, x.shape)
    dx = X.derivative(t).squeeze(0)

    # Plot interpolation and derivative
    fig, axs = plt.subplots(2, 1)
    plt.sca(axs[0])
    plt.plot(t, x)
    for time in X.grid_points:
        plt.axvline(time, linestyle='-.', alpha=0.5, linewidth=0.8, color='gray')
    plt.ylabel('X')
    plt.sca(axs[1])
    plt.plot(t, dx)
    for time in X.grid_points:
        plt.axvline(time, linestyle='-.', alpha=0.5, linewidth=0.8, color='gray')
    plt.ylabel('dX/dt')
    fig.suptitle(f'{dataset.capitalize()} set sample {n}: {interpolation_method} data interpolation')
    #plt.show()
    return fig


# Classes for creating the Neural CDE system
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_units, num_hidden_layers, activation_func='relu', layer_norm=False):
        ''' input_channels are the features in the data and hidden channels
            is an hyperparameter determining the dimensionality of the hidden state z'''
        super(CDEFunc, self).__init__()
        self.nfe = 0 # number of function evaluations
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.layer_norm = layer_norm
        if activation_func == 'celu': 
            self.activation = torch.nn.CELU()
        else:
            self.activation = torch.nn.ReLU()
        
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_units)
        self.linear_out = torch.nn.Linear(hidden_units, input_channels * hidden_channels)
        #self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden_layers - 1))
        layers = []
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_units, hidden_units))
            if self.layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_units))
            layers.append(self.activation)
        self.sequential = torch.nn.Sequential(*layers)
        if self.layer_norm:
            self.ln_in = torch.nn.LayerNorm(hidden_units) 
            self.ln_out = torch.nn.LayerNorm(input_channels * hidden_channels)

    def forward(self, t, z):
        self.nfe += 1
        z = self.linear_in(z)
        if self.layer_norm:
            z = self.ln_in(z)
        z = self.activation(z)
        z = self.sequential(z)
        z = self.linear_out(z)
        if self.layer_norm:
            z = self.ln_out(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels) # batch is first input
        z = z.tanh() # authors say that a final tanh non-linearity gives the best results
        return z

    def reset_nfe(self):
        self.nfe = 0


class NeuralCDE(torch.nn.Module):
    def __init__(self, vector_field, input_channels, hidden_channels, output_channels, interpolation_method, rtol=1e-4, atol=1e-6, layer_norm=False, seminorm=False):
        super(NeuralCDE, self).__init__()
        self.func = vector_field
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation_method = interpolation_method
        self.rtol = rtol
        self.atol = atol
        self.layer_norm = layer_norm
        self.seminorm = seminorm
        if self.layer_norm:
            self.ln_initial = torch.nn.LayerNorm(hidden_channels)
            self.ln_output = torch.nn.LayerNorm(output_channels)
    
    @property
    def nfe(self):
        return self.func.nfe

    def reset_nfe(self):
        self.func.reset_nfe()

    def forward(self, coeffs, times):
        X, cdeint_options = build_data_path(coeffs, times, self.interpolation_method)
        z0 = self.initial(X.evaluate(X.interval[0])) # initial hidden state must be a function of the first observation

        if self.seminorm:
            adjoint_options = cdeint_options.copy()
            adjoint_options['norm'] = 'seminorm'
        else:
            adjoint_options = {}

        if self.layer_norm:
            z0 = self.ln_initial(z0)
        z_t = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval, options=cdeint_options, adjoint_options=adjoint_options, rtol=self.rtol, atol=self.atol) # t=times[[0, -1]] is the same (but only when times is not None...)

        # Both z0 and z_T are returned from cdeint, extract just last value
        z_T = z_t[:, -1]
        pred_y = self.readout(z_T)
        if self.layer_norm:
            pred_y = self.ln_output(pred_y)        
        pred_y = torch.nn.functional.softmax(pred_y, dim=-1) # New. Added a soft-max to get soft assignments that work with the multi-class cross entropy loss function
        return pred_y


class NeuralCDE_stacked(torch.nn.Module):
    def __init__(self, vector_field1, vector_field2, input_channels, hidden_channels, output_channels, interpolation_method, rtol=1e-4, atol=1e-6, layer_norm=False, seminorm=False):
        super(NeuralCDE_stacked, self).__init__()
        self.func1 = vector_field1
        self.func2 = vector_field2
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation_method = interpolation_method
        self.rtol = rtol
        self.atol = atol
        self.layer_norm = layer_norm
        self.seminorm = seminorm
        if self.layer_norm:
            self.ln_initial = torch.nn.LayerNorm(hidden_channels)
            self.ln_output = torch.nn.LayerNorm(output_channels)
    
    @property
    def nfe(self):
        return self.func1.nfe + self.func2.nfe

    def reset_nfe(self):
        self.func1.reset_nfe()
        self.func2.reset_nfe()

    def forward(self, coeffs, times):
        X, cdeint_options1 = build_data_path(coeffs, times, self.interpolation_method)
        z0 = self.initial(X.evaluate(X.interval[0])) # initial hidden state must be a function of the first observation

        if self.seminorm:
            adjoint_options1 = cdeint_options1.copy()
            adjoint_options1['norm'] = 'seminorm'
        else:
            adjoint_options1 = {}

        if self.layer_norm:
            z0 = self.ln_initial(z0)
        z_t = torchcde.cdeint(X=X, z0=z0, func=self.func1, t=X.grid_points, options=cdeint_options1, adjoint_options=adjoint_options1, rtol=self.rtol, atol=self.atol) # t=times[[0, -1]] is the same (but only when times is not None...)

        # Intepolate output of firt solver and pass to second solver    
        z_coeffs = torchcde.linear_interpolation_coeffs(z_t)
        Z = torchcde.LinearInterpolation(z_coeffs)
        z0_ = Z.evaluate(Z.interval[0])
        cdeint_options2 = dict(jump_t=Z.grid_points)
        if self.seminorm:
            adjoint_options2 = cdeint_options2.copy()
            adjoint_options2['norm'] = 'seminorm'
        else:
            adjoint_options2 = {}
        z_t = torchcde.cdeint(X=Z, z0=z0_, func=self.func2, t=Z.interval, options=cdeint_options2, adjoint_options=adjoint_options2, rtol=self.rtol, atol=self.atol)

        # Both z0 and z_T are returned from cdeint, extract just last value
        z_T = z_t[:, -1]
        pred_y = self.readout(z_T)
        if self.layer_norm:
            pred_y = self.ln_output(pred_y)        
        pred_y = torch.nn.functional.softmax(pred_y, dim=-1) # New. Added a soft-max to get soft assignments that work with the multi-class cross entropy loss function
        return pred_y


class RNN(torch.nn.Module): # Thank you Python Engineer! https://www.youtube.com/watch?v=0_PgWWmauHk
    def __init__(self, input_channels, hidden_channels, num_layers, output_channels):
        super(RNN, self).__init__()
        self.num_layers = num_layers # how many RNN layers stacked together
        self.hidden_channels = hidden_channels
        self.rnn = torch.nn.RNN(input_channels, hidden_channels, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)
    
    def forward(self, x, times):
        # TODO: remove need of passing times and interpolation methods, they are meant for ncde model!
        # x -> batch_size, seq_length, input_size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)

        # Push through rnn (out: batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, h0)
        # Take only hidden vector of last sequence
        out = out[:, -1, :]
        # Decode into classes
        out = self.linear(out)
        out = torch.nn.functional.softmax(out, dim=-1) # Last layer same as ncde
        return out


class GRU(torch.nn.Module): # Thank you Python Engineer! https://www.youtube.com/watch?v=0_PgWWmauHk
    def __init__(self, input_channels, hidden_channels, num_layers, output_channels):
        super(GRU, self).__init__()
        self.num_layers = num_layers # how many RNN layers stacked together
        self.hidden_channels = hidden_channels
        self.gru = torch.nn.GRU(input_channels, hidden_channels, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)
    
    def forward(self, x, times):
        # TODO: remove need of passing times, they are meant for ncde model!
        # x -> batch_size, seq_length, input_size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)

        # Push through rnn (out: batch_size, seq_length, hidden_size)
        out, _ = self.gru(x, h0)
        # Take only hidden vector of last sequence
        out = out[:, -1, :]
        # Decode into classes
        out = self.linear(out)
        out = torch.nn.functional.softmax(out, dim=-1) # Last layer same as ncde
        return out


class LSTM(torch.nn.Module): # Thank you Python Engineer! https://www.youtube.com/watch?v=0_PgWWmauHk
    def __init__(self, input_channels, hidden_channels, num_layers, output_channels):
        super(LSTM, self).__init__()
        self.num_layers = num_layers # how many RNN layers stacked together
        self.hidden_channels = hidden_channels
        self.lstm = torch.nn.LSTM(input_channels, hidden_channels, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)
    
    def forward(self, x, times):
        # TODO: remove need of passing times and interpolation methods, they are meant for ncde model!
        # x -> batch_size, seq_length, input_size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)

        # Push through rnn (out: batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Take only hidden vector of last sequence
        out = out[:, -1, :]
        # Decode into classes
        out = self.linear(out)
        out = torch.nn.functional.softmax(out, dim=-1) # Last layer same as ncde
        return out


class ODERNNFunc(torch.nn.Module): # From Kidger 2020
    def __init__(self, hidden_channels, hidden_units, num_hidden_layers):
        super(ODERNNFunc, self).__init__()
        self.nfe = 0 # number of function evaluations
        layers = [torch.nn.Linear(hidden_channels, hidden_units)]
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_units, hidden_units))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_units, hidden_channels))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, t, z):
        self.nfe += 1
        return self.sequential(z)
    
    def reset_nfe(self):
        self.nfe = 0


class ODERNN(torch.nn.Module): # From Kidger 2020
    def __init__(self, input_channels, hidden_channels, hidden_units, num_hidden_layers, output_channels, rtol=1e-4, atol=1e-5):
        super(ODERNN, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.rtol = rtol
        self.atol = atol
        self.func = ODERNNFunc(hidden_channels, hidden_units, num_hidden_layers)
        self.gru = torch.nn.GRUCell(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)
    
    @property
    def nfe(self):
        return self.func.nfe

    def reset_nfe(self):
        self.func.reset_nfe()
    
    def _step(self, xi, h):
        new_h = self.gru(xi, h)
        h = new_h
        return h

    def evolve(self, h, time_diff):
        t = torch.Tensor([0, time_diff.item()]).to(time_diff.device)
        out = torchdiffeq.odeint_adjoint(func=self.func, y0=h, t=t, rtol=self.rtol, atol=self.atol)
        return out[1]

    def forward(self, x, times):
        # Initialite z0
        batch_dims = x.size(0)
        h0 = torch.zeros(batch_dims, self.hidden_channels, dtype=x.dtype, device=x.device)

        # Make first update
        x_unbound = x.unbind(dim=1)
        h = self._step(x_unbound[0], h0)

        # Evolve-updates steps
        hs = [h]
        time_diffs = times[1:] - times[:-1]
        for time_diff, xi in zip(time_diffs, x_unbound[1:]):
            h = self.evolve(h, time_diff)
            h = self._step(xi, h)
            hs.append(h)
        out = torch.stack(hs, dim=1)

        # Decode into classes
        out = out[:, -1, :] # take only last hidden vector
        out = self.linear(out)
        out = torch.nn.functional.softmax(out, dim=-1) # Last layer same as ncde
        return out


class CustomDataset(torch.utils.data.Dataset): # TODO: hdf5 dataset if this is what makes the FasterDataLoader faster
    def __init__(self, coeffs, labels):
        assert isinstance(coeffs, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass

    
class FastTensorDataLoader: # Edited from from Nando's class
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, dataset, batch_size=600, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *dataset: hdf5 dataset. Eg. Crop dataset
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an iterator is created out of this object.
            Recommendation: set shuffle to False, the underlying hd5y is than more efficient,
            because id can make use of the contiguous blocks of data.

        :returns: A FastTensorDataLoader.
        """
        self.dataset = dataset
        self.coeffs = dataset.tensors[0]
        self.labels = dataset.tensors[1]
        
        self.dataset_len = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0: 
            n_batches += 1 # what hapens to the last one => not full right?
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len: #start from beginning again # new epoch??
            raise StopIteration 
            self.i = 0
        if self.indices is not None:
            indices = np.sort(self.indices[self.i:self.i+self.batch_size])
            coeffs = self.coeffs[indices]
            labels = self.labels[indices]
        else:
            coeffs = self.coeffs[self.i:self.i+self.batch_size]
            labels = self.labels[self.i:self.i+self.batch_size]               
        self.i += self.batch_size
        return coeffs, labels

    def __len__(self):
        return self.n_batches


class _TensorEncoder(json.JSONEncoder): # From Patrick Kigger, to serialize also tensors and ndarrays when saving experiment output in a json file.
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def init_network_weights(net, std = 0.1): # From Nando. Not used, Pytorch default initialization of layers seems alright.
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=std)
            torch.nn.init.constant_(m.bias, val=0)


def count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


def add_weight_regularisation(loss_fn, model, scaling=0.005):
    """ Adds L2-regularisation to the loss function """
    def new_loss_fn(y_true, y_pred):
        total_loss = loss_fn(y_true, y_pred)
        for parameter in model.parameters():
            if parameter.requires_grad:
                total_loss += scaling * parameter.norm()
        return total_loss
    return new_loss_fn


def compute_multiclass_cross_entropy(y_true, y_pred): # From Nando
    eps = 1e-10
    ce_loss = -(y_true * torch.log(y_pred + eps)).sum(dim=1).mean()  # because torch.nn.functional.cross_entropy -> multi-target not supported
    return ce_loss


def compute_multiclass_cross_entropy2(y_true, y_pred): # REMEMBER: if using this, comment or remove final softmax on each model!. Also note that this loss_fn doesn't take advantage of labels not being always hard one-hot
    y_true_thresholded = torch.argmax(y_true, dim=1) # because of --> RuntimeError: 1D target tensor expected, multi-target not supported
    ce_loss = torch.nn.functional.cross_entropy(y_pred, y_true_thresholded) 
    return ce_loss


def compute_total_matches(y_true, y_pred):
    thresholded_pred = torch.argmax(y_pred, dim=1)
    labels_max_indices = torch.argmax(y_true, dim=1)
    prediction_matches = (thresholded_pred == labels_max_indices).to(y_pred.dtype).sum()
    return prediction_matches


def compute_f1_score(y_true, y_pred):
    thresholded_pred = torch.argmax(y_pred, dim=1).detach().cpu()
    labels_max_indices = torch.argmax(y_true, dim=1).detach().cpu()
    f1_score = sklearn.metrics.f1_score(labels_max_indices, thresholded_pred, average=None) # returns per-class F1 scores, taking the mean of this is eq. to average='macro'
    return f1_score


def compute_confusion_matrix(y_true, y_pred, class_names=None):
    thresholded_pred = torch.argmax(y_pred, dim=1).detach().cpu()
    labels_max_indices = torch.argmax(y_true, dim=1).detach().cpu()
    confusion = sklearn.metrics.confusion_matrix(labels_max_indices, thresholded_pred, labels=class_names, normalize='true')
    return confusion


def plot_confusion_matrix(confusion, labels, expID):
    df_cm = pd.DataFrame(confusion, index = [i for i in labels], columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=False, cmap='Blues', vmin=0, vmax=1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=60)
    plt.tight_layout()
    cm_path = os.path.join(here, 'figures/confusion')
    if not os.path.exists(cm_path):
        os.makedirs(cm_path, exist_ok=True)
    print(f'Saving confusion matrix in: {cm_path}')
    plt.savefig(f'{cm_path}/cm_exp{expID}.png', bbox_inches='tight')
    #plt.show()


def evaluate_metrics(dataloader, model, times, device, loss_fn=compute_multiclass_cross_entropy, labels=None):
    with torch.no_grad():
        total_dataset_size = 0
        total_loss = 0
        total_accuracy = 0
        total_f1_score = 0
        batch_y_all = []
        pred_y_all = []
    
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            batch_data, batch_y = batch
            batch_size = batch_y.size(0)
            pred_y = model(batch_data, times).squeeze(-1)

            # Add metrics of current batch
            total_dataset_size += batch_size
            total_loss += loss_fn(batch_y, pred_y) * batch_size   # assuming mean reduction in loss function
            total_accuracy += compute_total_matches(batch_y, pred_y)
            batch_y_all.append(batch_y)
            pred_y_all.append(pred_y)

        # Average metrics over the whole dataset
        total_loss /= total_dataset_size
        total_accuracy /= total_dataset_size

        # F1 score (must compute on all labels and predictions at once)
        batch_y_all = torch.cat(batch_y_all, dim=0)
        pred_y_all = torch.cat(pred_y_all, dim=0)
        total_f1_score = compute_f1_score(batch_y_all, pred_y_all)
        confusion = compute_confusion_matrix(batch_y_all, pred_y_all, labels)
        
        # Return metrics
        metrics = dict(dataset_size=total_dataset_size, loss=round(total_loss.item(), 5), accuracy=round(total_accuracy.item(), 5), f1_score=round(total_f1_score.mean().item(), 5), perclass_f1= total_f1_score.round(5), confusion=confusion.round(2))
        return metrics


def grid_search_runs(args, repeats=1):
    ''' Makes a grid with every combination of the arguments that are passed as a list
        and one run for each combination.'''
    def combinations(*args):
        combinations_list = []
        for combination in itertools.product(*args):
            combinations_list.append(combination)
        return combinations_list  
    grid = []
    for key, value in vars(args).items():
        if isinstance(value, list):
            grid.append((key, value))
    keys = [param[0] for param in grid]
    values = [param[1] for param in grid]
    combinations_list = combinations(*values)
    print('Grid search combinations:', keys, combinations_list)
    original_save = args.save
    if combinations_list is not None:
        for run, element in enumerate(combinations_list):
            args_new_dict = vars(args)
            print(f'\nGrid search combination {run + 1}/{len(combinations_list)}:')
            subfolder_path = '/'
            for i, key in enumerate(keys):
                print(f'{key}={element[i]}')
                args_new_dict[key] = element[i]
                subfolder_path = subfolder_path + key + str(element[i])
            args_new_dict['save'] = original_save + subfolder_path                
            args_new_dict['grid_search'] = None
            args_new = argparse.Namespace(**args_new_dict)
            combination_start_time = time.time()
            for i in range(repeats):
                repeat_start_time = time.time()
                if repeats > 1: print(f'\nRepeat: {i + 1}/{repeats}')
                main(args_new)
                if repeats > 1: print('Repeat run time:', round(time.time() - repeat_start_time, 2), 'seconds')
            print('Combination run time:', round(time.time() - combination_start_time, 2), 'seconds')
        print(f'Total grid search run (n of combinations={len(combinations_list)}) time: {round(time.time() - start_time, 2)} seconds')


def parse_args():
    parser = argparse.ArgumentParser()	
    parser.add_argument("--data_root", type=str, default="C:\\Users\\jukin\\Desktop\\Ms_Thesis\\Data\\Crops\\processed", help='[default=%(default)s].')
    parser.add_argument("--noskip", default=False, action="store_true", help='Activate for using even cloudy observations [default=%(default)s].')
    parser.add_argument("--reduced", default=True, action="store_true", help='Use only a fraction of the dataset features (for Crops dataset) [default=%(default)s].')
    parser.add_argument("--intensity", default=True, action="store_true", help='Äctivate for using appending observational mask for every channel as extra features [default=%(default)s].')
    parser.add_argument("--time_default", default=False, action="store_true", help='To not pass the original dataset timestamps to the interpolation and model and use the default equally spaced time array of torchde interpolation methods [default=%(default)s].')
    parser.add_argument("--interpol_method", type=str, default='linear', choices=['cubic', 'linear', 'rectilinear', 'SEkernel'], help='Interpolation method to use for creating continous data path X [default=%(default)s].')
    parser.add_argument("--ntrain", type=int, default=None, help='Number of train samples [default=%(default)s].')
    parser.add_argument("--nval", type=int, default=None, help='Number of validation samples [default=%(default)s].')
    parser.add_argument("--max_epochs", type=int, default=40, help='Maximum number of epochs [default=%(default)s].')
    parser.add_argument("--lr", type=float, default=0.001, nargs='*', help='Optimizer initial learning rate [default=%(default)s].')
    parser.add_argument("--BS", type=int, default=512, nargs='*', help='Batch size for train, validation and test sets [default=%(default)s].')
    parser.add_argument("--num_workers", type=int, default=0, help='Num workers for dataloaders [default=%(default)s].')
    parser.add_argument("--HC", type=int, default=64, nargs='*', help='Hidden channels. Size of the hidden state in NCDE or RNN models [default=%(default)s].')
    parser.add_argument("--HL", type=int, default=1, nargs='*', help='Number of hidden layers in the vector field or of RNN layers if an RNN model is selected [default=%(default)s].')
    parser.add_argument("--HU", type=int, default=128, nargs='*', help='Number of hidden units in the vector field [default=%(default)s].')
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'celu'],  help='Intermediate activation function in vector field of NCDE (final one is always tanh) [default=%(default)s].')
    parser.add_argument("--layer_norm", default=False, action="store_true", help='Apply layer norm to before every activation function [default=%(default)s].')
    parser.add_argument("--ES_patience", type=int, default=5, help='Early stopping number of epochs to wait before stopping [default=%(default)s].')
    parser.add_argument("--lr_decay", default=True, action="store_true", help='Add learning rate decay if when no improvement of training accuracy [default=%(default)s].')
    parser.add_argument("--lr_decay_factor", type=float, default=0.1, nargs='*', help='Learning rate decay factor [default=%(default)s].')
    parser.add_argument("--regularization", type=float, default=None, help='If not None adds L2 regularization to the loss function with scaling specified by the argument [default=%(default)s].')
    parser.add_argument("--pin_memory", default=False, action="store_true", help='Pass pin memory option to torch.utils.data.DataLoader [default=%(default)s].')
    parser.add_argument("--save", type=str, default='results', help='Name of new or existing folder where to save results [default=%(default)s].')
    parser.add_argument("--resume", default=None, help='ID of experiment for resuming training. If None runs a new experiment [default=%(default)s].')
    parser.add_argument("--no_logwandb", default=False, action='store_true', help='Log the run in weights and biases [default=%(default)s].')		
    parser.add_argument("--fast_dataloader", default=False, action='store_true', help='Try out fast dataloader (with shuffle=False) [default=%(default)s].')		
    parser.add_argument("--grad_clip", type=float, default=None, help='Max norm to clip gradients to [default=%(default)s].')		
    parser.add_argument("--model", type=str, default='ncde', choices=['ncde', 'ncde_stacked', 'odernn', 'rnn', 'gru', 'lstm'], help='Model to use [default=%(default)s].')
    parser.add_argument("--seminorm", default=True, action="store_true", help='If to use seminorm for 2x speed up odeint adaptative solvers [default=%(default)s].')
    parser.add_argument("--rtol", type=float, default=1e-4, help='Relative tolerance for odeint solvers [default=%(default)s].')
    parser.add_argument("--atol", type=float, default=1e-6, help='Absolute tolerace for odeint solvers [default=%(default)s].')
    parser.add_argument("--grid_search", type=int, default=None, help='Number of repetitions for grid search. If passed and there is any argument as a list, then n repetitions will be made for every possible combination (list-argument options are: lr, BS, HC, HL, HU, lrdecay factor) [default=%(default)s].')
    args = parser.parse_args()
    return args


# Main code
def main(args):
    # Grid search over arguments that are a list
    if args.grid_search is not None:
        print('Original arguments:\n',args)
        grid_search_runs(args, repeats=args.grid_search)
        sys.exit()

    # Assign arguments
    args_dict = vars(args)
    grid_search = args_dict['grid_search']
    absolute_data_directory_path = args_dict['data_root']
    noskip= args_dict['noskip']
    reduced =  args_dict['reduced']
    intensity =  args_dict['intensity']
    time_default = args_dict['time_default']
    interpolation_method = args_dict['interpol_method']
    ntrain_samples = args_dict['ntrain']
    nval_samples = args_dict['nval']
    num_epochs = args_dict['max_epochs']
    learning_rate = args_dict['lr'] = args_dict['lr'][0] if isinstance(args_dict['lr'], list) else args_dict['lr']
    batch_size = args_dict['BS'] = args_dict['BS'][0] if isinstance(args_dict['BS'], list) else args_dict['BS']
    num_workers = args_dict['num_workers']
    hidden_channels = args_dict['HC'] = args_dict['HC'][0] if isinstance(args_dict['HC'], list) else args_dict['HC']
    num_hidden_layers = args_dict['HL'] = args_dict['HL'][0] if isinstance(args_dict['HL'], list) else args_dict['HL']
    hidden_units = args_dict['HU'] = args_dict['HU'][0] if isinstance(args_dict['HU'], list) else args_dict['HU']
    activation_func = args_dict['activation']
    layer_norm = args_dict['layer_norm']
    early_stopping_patience = args_dict['ES_patience']
    lr_decay = args_dict['lr_decay']
    lr_decay_factor = args_dict['lr_decay_factor'] = args_dict['lr_decay_factor'][0] if isinstance(args_dict['lr_decay_factor'], list) else args_dict['lr_decay_factor']
    l2_reg_scaling = args_dict['regularization']
    pin_memory = args_dict['pin_memory']
    results_folder = args_dict['save']
    checkpoint_expID = args_dict['resume']
    logwandb = not args_dict['no_logwandb']
    fast_loader = args_dict['fast_dataloader']
    grad_clip = args_dict['grad_clip']
    use_model = args_dict['model']
    seminorm = args_dict['seminorm']
    rtol = args_dict['rtol']
    atol = args_dict['atol']
    
    # Logger
    script_name = __file__.split('/')[-1].split('.')[0]
    if logwandb:
        wandb.init(config=args_dict, project='crops_gpu_3.0', save_code=True)
        expID = wandb.run.id
        wandb.run.name = expID
        wandb.run.save()
    else:
        expID = int(random.SystemRandom().random()*1e7)
    print(f'Experiment {expID}')
    print(f'Arguments: \n {args_dict}')

    # Get data and labels
    data = get_data(absolute_data_directory_path=absolute_data_directory_path, use_noskip=noskip, reduced=reduced, ntrain=ntrain_samples, nval=nval_samples, use_model=use_model, intensity=intensity)
    times = None if time_default and (use_model == 'ncde' or use_model == 'ncde_stacked') else data['times']
    intensity_str = '_intensity' if intensity else ''
    coeffs_directory = os.path.join(absolute_data_directory_path, f'interpolation_coefficients{intensity_str}')
    if use_model == 'ncde' or use_model == 'ncde_stacked':
        coefficients = get_interpolation_coeffs(coeffs_directory, data, times, interpolation_method=interpolation_method, use_noskip=noskip, reduced=reduced)
        fig = plot_interpolation_path(coefficients, 'train', times, interpolation_method)
        if logwandb:
            wandb.log({'Interpolation sample': fig})

    # Define parametes
    input_channels = data['train_data'].size(2) # 54 features + 1 for time
    output_channels = data['train_labels'].size(1) # 19 field classes
    hidden_channels = hidden_channels  # dimension of the hidden state z
    hidden_units = hidden_units # vector field network layers size
    num_hidden_layers = num_hidden_layers # vector field network number of layers

    # Use GPU if available
    print('GPU devices available:', torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Define model
    if use_model == 'ncde':
        vector_field = CDEFunc(input_channels, hidden_channels, hidden_units, num_hidden_layers, activation_func, layer_norm=layer_norm)
        model = NeuralCDE(vector_field=vector_field, input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels, interpolation_method=interpolation_method, layer_norm=layer_norm, seminorm=seminorm, rtol=rtol, atol=atol).to(device)
    elif use_model == 'ncde_stacked':
        vector_field1 = CDEFunc(input_channels, hidden_channels, hidden_units, num_hidden_layers, activation_func, layer_norm=layer_norm)
        vector_field2 = CDEFunc(hidden_channels, hidden_channels, hidden_units, num_hidden_layers, activation_func, layer_norm=layer_norm)
        model = NeuralCDE_stacked(vector_field1=vector_field1, vector_field2=vector_field2, input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels, interpolation_method=interpolation_method, layer_norm=layer_norm, seminorm=seminorm, rtol=rtol, atol=atol).to(device)
    elif use_model == 'rnn':
        model = RNN(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_hidden_layers, output_channels=output_channels).to(device)
    elif use_model == 'gru':
        model = GRU(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_hidden_layers, output_channels=output_channels).to(device)
    elif use_model == 'lstm':
        model = LSTM(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_hidden_layers, output_channels=output_channels).to(device)
    elif use_model =='odernn':
        model = ODERNN(input_channels=input_channels, hidden_channels=hidden_channels, hidden_units=hidden_units, num_hidden_layers=num_hidden_layers, output_channels=output_channels, rtol=rtol, atol=atol).to(device)
    
    print(model)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = compute_multiclass_cross_entropy

    # Add learning rate decay
    if lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=lr_decay_factor, mode='max', verbose=True)

    # Add L2 regularization to loss function
    if l2_reg_scaling is not None:
        loss_fn = add_weight_regularisation(loss_fn, model, scaling=l2_reg_scaling)

    # Track models gradients with logger
    if logwandb:
        wandb.watch(model, criterion=loss_fn, log='all')

    # Load state from checkpoint
    checkpoints_path = os.path.join(here, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    if checkpoint_expID is not None:
        checkpoint = torch.load(os.path.join(checkpoints_path, f'exp_{checkpoint_expID}_checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        checkpoint_best_epoch = checkpoint['epoch']
        checkpoint_args_dict = checkpoint['args_dict']
        checkpoint_best_val_loss = checkpoint['val_loss']
        print(f'Loaded checkpoint from experiment {checkpoint_expID}, last computed epoch was {checkpoint_best_epoch} with val loss of {checkpoint_best_val_loss}.')
        print(f'Original run arguments: \n {checkpoint_args_dict}')
    print(f'Model in: {next(model.parameters()).device}')

    # Define torch dataset object
    if use_model == 'ncde' or use_model == 'ncde_stacked':
        train_TensorDataset = torch.utils.data.TensorDataset(coefficients['train_coeffs'], data['train_labels'])
        val_TensorDataset = torch.utils.data.TensorDataset(coefficients['val_coeffs'], data['val_labels'])
        test_TensorDataset = torch.utils.data.TensorDataset(coefficients['test_coeffs'], data['test_labels'])
    else:
        train_TensorDataset = torch.utils.data.TensorDataset(data['train_data'], data['train_labels'])
        val_TensorDataset = torch.utils.data.TensorDataset(data['val_data'], data['val_labels'])
        test_TensorDataset = torch.utils.data.TensorDataset(data['test_data'], data['test_labels'])
    
    # Define torch dataloader object
    if fast_loader:
        train_dataloader = FastTensorDataLoader(train_TensorDataset, batch_size=batch_size, shuffle=False)
        val_dataloader = FastTensorDataLoader(val_TensorDataset, batch_size=batch_size,)
        test_dataloader = FastTensorDataLoader(test_TensorDataset, batch_size=batch_size)
    else: 
        train_dataloader = torch.utils.data.DataLoader(train_TensorDataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_TensorDataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        test_dataloader = torch.utils.data.DataLoader(test_TensorDataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Store intended output in a list and update logger
    output = {}
    config = dict(script_name=script_name, expID=expID, num_parameters=count_parameters(model))
    if logwandb:
        wandb.config.update(config)
    config.update(args_dict)
    output['config'] = config

    # Training and validation loop
    best_val_f1 = 0
    best_val_f1_epoch = 0
    best_val_acc = 0
    history = []
    current_lr = learning_rate
    is_ode = use_model in ('ncde', 'ncde_stacked', 'odernn')

    print('Learning...')
    if not time_default: times = times.to(device) # Shouldn't give problems because times is None otherwise.
    start = time.time()
    pbar = tqdm.tqdm(range(1, num_epochs + 1))
    n = 0 # num of batch iterations counter
    for epoch in pbar:
        start_epoch = time.time()
        epoch_nfes = 0 # number of fuction evaluations for ncde and odernn on forward pass
        epoch_bnfes = 0 # number of fuction evaluations for ncde and odernn on backward pass
        model.train()
        for batch in train_dataloader:
            # Forward pass
            if is_ode:
                model.reset_nfe()
            optimizer.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            batch_data, batch_y = batch # labels are not all 1s and 0s -> Nando leverages this soft labels for training and for accuracy he does with hard labels (argmax)
            pred_y = model(batch_data, times).squeeze(-1)
            loss = loss_fn(batch_y, pred_y) # problem: returns nan because cannot do log of a negative value (prediction) -> solved it with softmax!
            
            if is_ode:
                iteration_nfes = model.nfe
                epoch_nfes += iteration_nfes
                model.reset_nfe()

            # Backward pass
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=grad_clip) # New
            optimizer.step()

            if is_ode:
                iteration_bnfes = model.nfe
                epoch_bnfes += iteration_bnfes
                model.reset_nfe()

            # Log gradients for inspection
            for name, parameter in model.named_parameters():
                if logwandb:
                    wandb.log({f'gradient/{name}': wandb.Histogram(parameter.grad.detach().cpu().numpy()), 'iteration': n}) 
            
            if logwandb:
                wandb.log({'batch training loss': loss, 'iteration': n})
            n += 1

        # Compute traininig and validation metrics for the epoch
        model.eval()
        train_metrics = evaluate_metrics(train_dataloader, model, times, device, loss_fn=loss_fn)	# TODO: check that is doesn't slow down everything too much.	
        val_metrics = evaluate_metrics(val_dataloader, model, times, device, loss_fn=loss_fn) # TODO: check that is doesn't slow down everything too much.
        epoch_time = round(time.time() - start_epoch, 1)
        total_nfes = epoch_nfes + epoch_bnfes
        pbar.write(f'Epoch: {epoch}, lr={current_lr}, epoch time: {epoch_time}s, nfes: {epoch_nfes}, bnfes: {epoch_bnfes}, total nfes: {total_nfes}  || Training loss: {train_metrics["loss"]: 0.5f}  | Training accuracy: {train_metrics["accuracy"]: 0.5f} | Training F1-score: {train_metrics["f1_score"]: 0.5f}  || Validation loss: {val_metrics["loss"]: 0.5f}  | Validation accuracy: {val_metrics["accuracy"]: 0.5f} | Validation F1-score: {val_metrics["f1_score"]: 0.5f}')
        history.append(dict(epoch=epoch, lr=current_lr, epoch_time=epoch_time, nfes=epoch_nfes, bnfes=epoch_bnfes, total_nfes=total_nfes, train_metrics=train_metrics, val_metrics=val_metrics))

        # Log train and val metrics
        if logwandb:
            wandb.log({'training loss': train_metrics["loss"], 'training accuracy': train_metrics["accuracy"], 'training F1-score': train_metrics["f1_score"],
                    'validation loss': val_metrics["loss"], 'validation accuracy': val_metrics["accuracy"], 'validation F1-score': val_metrics["f1_score"],
                    'epoch': epoch, 'epoch time': epoch_time,'nfes': epoch_nfes, 'bnfes': epoch_bnfes, 'total nfes': total_nfes})

        # Update learning rate
        if lr_decay:
            scheduler.step(val_metrics['f1_score']) # if metric is loss instead, remember to change mode of optimizer to 'min' (default).
            current_lr = scheduler._last_lr[0]	

        # Early stopping and checkpointing
        if val_metrics["f1_score"] >= best_val_f1 * 1.0001:
            best_val_f1 = val_metrics["f1_score"]
            best_val_f1_epoch = epoch
            best_val_acc = val_metrics["accuracy"]
            best_model = copy.deepcopy(model)
            
            # Save checkpoint of the model # TODO: check that is doesn't slow down everything too much.
            checkpoint_path = os.path.join(checkpoints_path, f'exp_{expID}_checkpoint.pt')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_metrics["loss"],
            'args_dict': args_dict
            }, checkpoint_path)
            
        elif (early_stopping_patience is not None) and (epoch >= best_val_f1_epoch + early_stopping_patience):
            pbar.write(f'Breaking because of no improvement in validation F1-score after {early_stopping_patience} epochs.')
            break

    train_time = round(time.time() - start, 1)
    output.update(dict(history=history))
    print(f'Train time: {train_time} seconds')
    if logwandb:
        wandb.log({'train time': train_time})
  
    # Evaluate performance on test dataset (attention for TUM dataset: this is a very small dataset and is missing 2 classes)
    print('Testing...')
    model.eval()
    test_metrics_best_model = evaluate_metrics(test_dataloader, best_model, times, device, loss_fn=loss_fn)
    best_epoch = best_val_f1_epoch
    test_acc = test_metrics_best_model["accuracy"] 
    test_f1 = test_metrics_best_model["f1_score"]
    print(f'Best epoch: {best_epoch} | best val accuracy: {best_val_acc}, best val f1: {best_val_f1}, test accuracy best model: {test_acc}, test f1 score best model: {test_f1}')

    # Log test metrics
    if logwandb:
        wandb.log({'best epoch': best_epoch, 'best val acc': best_val_acc, 'best val f1': best_val_f1, 'test accuracy (w/best model)': test_acc, 'test F1-score (w/best model)': test_f1})

    output.update(dict(test_metrics=dict(best_epoch=best_epoch, best_val_acc=best_val_acc, best_val_f1=best_val_f1, test_accuracy=test_acc, test_f1_score=test_f1, train_time=train_time)))
    
    # Plot confusion matrix on best results on validation set
    dataset = 'TUM'
    labels = TUM_labels_names if dataset == 'TUM' else swisscrop_labels_names
    val_metrics_best_model = evaluate_metrics(val_dataloader, best_model, times, device, loss_fn=loss_fn)    
    plot_confusion_matrix(confusion=val_metrics_best_model["confusion"], labels=labels, expID=expID)
    if logwandb:
        values = val_metrics_best_model["perclass_f1"]
        labels = labels
        data = [[label, val] for (label, val) in zip(labels, values)]
        table = wandb.Table(data=data, columns = ["class name", "f1-score"])
        wandb.log({"best validation F1-scores" : wandb.plot.bar(table, "class name", "f1-score", title="Best validation per class F1-scores")})
    
    # Save output
    noskip = 'noskip' if noskip else ''
    red = 'red' if reduced else ''
    timing = 'eqspaced' if times is None or interpolation_method == 'rectilinear' else 'irrspaced'
    results_path = os.path.join(here, results_folder)
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    n = 0
    while glob.glob(os.path.join(results_path, f'*results{n}*_{batch_size}BS_{learning_rate}lr_{lr_decay_factor}lrdecay_{hidden_channels}HC_{num_hidden_layers}HL_{hidden_units}HU_{interpolation_method}_{timing}{noskip}{red}.json')):
        n += 1
    with open(os.path.join(results_path, f'results{n}_{expID}_{batch_size}BS_{learning_rate}lr_{lr_decay_factor}lrdecay_{hidden_channels}HC_{num_hidden_layers}HL_{hidden_units}HU_{interpolation_method}_{timing}{noskip}{red}.json'), 'w') as f:
        json.dump(output, f, cls=_TensorEncoder)

if __name__ == '__main__':
    start_time = time.time()
    main(args=parse_args())  
    print('Run time', round(time.time() - start_time, 2), 'seconds')


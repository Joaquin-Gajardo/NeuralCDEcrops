"""
Created on Fri Oct 23 2020
@author: jgajardo
Code structure and snippets borrowed from Patrick Kridger and Nando Metzger.
Versions:
v1: 04.11.2020. Added F1 score, evaluate metrics functions, clearer output.
v2: 06.11.2020. Added early stopping, modified architecture, L2 regularization, named experiments output, modified get data function to include clouds mask, added some helper functions.
v3: 09.11.2020. Added argparse, modified early stopping conditions.
v4: 11.09.2020. LR as an argument and LR decay, added wandb logger. 16.11.2020 lr decay as an option and improved save text file name format, added samples option. 18.11.2020 Added num workers. Changed eps to 10^-10. 20.11.2020. Added memory pinning.
v5: 23.11.2020. Added histogram for gradients in wandb.
v6: 07.12.2020. -Added experiment ID and save and load checkpoints. Using noskip datasets and return mask too in get_data function. Log gradients
                -Modified data preprocessing considerably and interpolation, added option for reduced dataset, times to use, interpolation method and save coeffs as dataset. Plot sample function, build interpolation path function.
v7: 09.12.2020. Added batch norm and root data path argument. Added tqdm progress bar for measuring epochs speed. Added faster dataloader option.
v8: 14.12.2020. Added gradient clipping and replaced batch norm by layer norm options. 15.12.2020. Added RNN baseline models and semilog speed up option. 23.12. Added odernn baseline, atol/rtol options and nfes + time logging. 24.12. Changed results save name and argument. 29.12. Added activation functions, minor changes to defaults for lr decay and grid search option. 05.01. Change in test metrics best model and to regularization argument.
"""

# Import libraries
import numpy as np
import math
import torch
import torchdiffeq
import torchcde
import matplotlib.pyplot as plt
import wandb
import h5py
import argparse
import sklearn.metrics
import copy
import time
import os
import sys
import glob
import pathlib
import warnings
import random
import cProfile
import tqdm
import pdb
from itertools import product

# Check path
here = pathlib.Path(__file__).resolve().parent

# Ignore warnings
warnings.filterwarnings("ignore")

################################################################################################
# Swiscrop labels
swisscrop_labels_names = ['No Label','Maize', 'Meadow', 'Pasture', 'Potatoes', 'Spelt', 'Sugarbeets', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat', 'Winter barley', 'Winter rapeseed', 'Winter wheat']
swisscrop_labels = [0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]

#make label Tags
TUM_labels_names = [ "other", "corn", "meadow", "asparagus", "rape", "hop", "summer oats", "winter spelt", "fallow", "winter wheat",
                    "winter barley", "winter rye", "beans", "winter triticale", "summer barley", "peas", "potatoe", "soybeans", "sugar beets" ]
TUM_labels_dict = {k: i for i, k in enumerate(TUM_labels_names)}
reverse_TUM_labels_dict = {v: k for k, v in TUM_labels_dict.items()}

# Get processed data from TUM dataset (TODO add script with Nando's code of data processing!)
def get_data(absolute_data_directory_path, use_noskip=True, reduced=False, ntrain=None, nval=None, use_model='ncde'):
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

    # Reduce data features if required (only keep features of central pixel of the 3x3 neighbourhood)
    if reduced:
        data['train_data'] = data['train_data'][:, :, 4:-1:9]
        data['val_data'] = data['val_data'][:, :, 4:-1:9]
        data['test_data'] = data['test_data'][:, :, 4:-1:9]

        data['train_mask'] = data['train_mask'][:, :, 4:-1:9]
        data['val_mask'] = data['val_mask'][:, :, 4:-1:9]
        data['test_mask'] = data['test_mask'][:, :, 4:-1:9]

    # Impute NaNs where mask is 0 (non-observed pixel due to bad weather)	
    # Get mask data and transform to boolean to avoid matching numbers¨
    if use_model == 'ncde':
        train_mask = data['train_mask'].to(bool)
        val_mask = data['val_mask'].to(bool)
        test_mask = data['test_mask'].to(bool)
        
        # Impute NaNs in non-observed pixels (0=False=unobserved)
        data['train_data'][train_mask == False] = float('nan')
        data['val_data'][val_mask == False] = float('nan')
        data['test_data'][test_mask == False] = float('nan')

        # Append time as a feature at the beginning of last dimension (features dimension)
        for i in ['train_data', 'val_data', 'test_data']:
            t = data['times'].unsqueeze(0).repeat(data[i].size(0), 1).unsqueeze(-1)
            data[i] = torch.cat([t, data[i]], dim=-1)

        # Padd end of sequences with fill forward last observed row
        data['train_data'] = fill_last_nonnan_forward(data['train_data'])
        data['val_data'] = fill_last_nonnan_forward(data['val_data'])
        data['test_data'] = fill_last_nonnan_forward(data['test_data'])

        # Padd beginning of sequences with fill backward first observed row
        data['train_data'] = fill_first_nonnan_backward(data['train_data'])
        data['val_data'] = fill_first_nonnan_backward(data['val_data'])
        data['test_data'] = fill_first_nonnan_backward(data['test_data'])

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
    
    # Print missing values rate
    train_missing_rate = get_missing_values_rate(data['train_data'])
    val_missing_rate = get_missing_values_rate(data['val_data'])
    test_missing_rate = get_missing_values_rate(data['test_data'])
    print(f'Train data has {train_missing_rate * 100 :0.3f}% of missing values.')
    print(f'Val data has {val_missing_rate * 100 :0.3f}% of missing values.')
    print(f'Test data has {test_missing_rate * 100 :0.3f}% of missing values.')

    return data

def fill_last_nonnan_forward(x): # warning: it only works for tensors of ndim=3 and to fill in the dim=1.
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    tmp = np.flip(x, axis=1)
    idxs = np.argmax(~np.isnan(tmp), axis=1)
    idx = np.min(idxs[:, 1:], axis=1)
    for i, sample in enumerate(tmp):
        tmp[i, :idx[i], :] = np.tile(sample[idx[i], :], (idx[i], 1))
    x_mod = np.flip(tmp, axis=1)
    x_mod = torch.from_numpy(x)
    assert isinstance(x_mod, torch.Tensor)
    return x_mod

def fill_first_nonnan_backward(x): # warning: it only works for tensors of ndim=3 and to fill in the dim=1. Also inefficient because of for loop.
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    idxs = np.argmax(~np.isnan(x), axis=1)
    idx = np.min(idxs[:, 1:], axis=1)
    for i, sample in enumerate(x):
        x[i, :idx[i], :] = np.tile(sample[idx[i], :], (idx[i], 1))
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
        if interpolation_method == 'cubic':
            coefficients['train_coeffs'] = torchcde.natural_cubic_spline_coeffs(data['train_data'], t=times)
            coefficients['val_coeffs'] = torchcde.natural_cubic_spline_coeffs(data['val_data'], t=times)
            coefficients['test_coeffs'] = torchcde.natural_cubic_spline_coeffs(data['test_data'], t=times)
        
        elif interpolation_method == 'linear':
            coefficients['train_coeffs'] = torchcde.linear_interpolation_coeffs(data['train_data'], t=times)
            coefficients['val_coeffs'] = torchcde.linear_interpolation_coeffs(data['val_data'], t=times)
            coefficients['test_coeffs'] = torchcde.linear_interpolation_coeffs(data['test_data'], t=times)
        
        elif interpolation_method == 'rectilinear': # rectifilinear doesn't work when passing time argument
            if timing == 'irrspaced': print('Warning: will do default equally spaced time array instead, rectifilinear interpolation currently works with it only.')
            coefficients['train_coeffs'] = torchcde.linear_interpolation_coeffs(data['train_data'], rectilinear=0)
            coefficients['val_coeffs'] = torchcde.linear_interpolation_coeffs(data['val_data'], rectilinear=0)
            coefficients['test_coeffs'] = torchcde.linear_interpolation_coeffs(data['test_data'], rectilinear=0)
    
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
        print('Loading corresponding interpolation coefficients ...')
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
        cdeint_options = dict(grid_points=X.grid_points, eps=1e-5)

    elif interpolation_method == 'rectilinear': # rectifilinear doesn't work when passing time argument
        X = torchcde.LinearInterpolation(coeffs)
        cdeint_options = dict(grid_points=X.grid_points, eps=1e-5)

    return X, cdeint_options

def plot_interpolation_path(coefficients, dataset, times, interpolation_method, n=None):
    print('Plotting interpolation of a sample for sanity check...')
    coeffs = coefficients[f'{dataset}_coeffs']
    if n is None or n > coeffs.size(0):
        n = np.random.randint(0, coeffs.size(0))
    coeffs = coeffs[n]

    X = build_data_path(coeffs, times, interpolation_method)[0]
    t = None if times == '' else times
    if interpolation_method == 'rectilinear':        
        t = X.grid_points # if it's rectifilear (or linear) the ode solver will evaluate at gridpoints anyway.
        print('t knots:', t, t.shape)
    elif interpolation_method == 'cubic' or interpolation_method == 'linear': # plot more points to see the true shape of them (discontinuities in dX for linear and the smoothness of cubic) 
        t = np.linspace(0., X.interval[-1], 1001)

    print('t for plot:', t, t.shape)
    x = X.evaluate(t)
    print('sample x for plot:', x, x.shape)
    dx = X.derivative(t)

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
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, activation_func='relu', layer_norm=False):
        ''' input_channels are the features in the data and hidden channels
            is an hyperparameter determining the dimensionality of the hidden state z'''
        super(CDEFunc, self).__init__()
        self.nfe = 0 # number of function evaluations
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_in = torch.nn.LayerNorm(hidden_hidden_channels) 
            self.ln_out = torch.nn.LayerNorm(input_channels * hidden_channels)
        # consider to do activation in-place if memory usage is too intensive
        if activation_func == 'leaky':
            self.activation = torch.nn.LeakyReLU()
        elif activation_func == 'elu':
            self.activation = torch.nn.ELU()
        elif activation_func == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, t, z):
        self.nfe += 1
        z = self.linear_in(z)
        if self.layer_norm:
            z = self.ln_in(z)
        z = self.activation(z)
        for linear in self.linears:
            z = linear(z)
            if self.layer_norm:
                z = self.ln_in(z)
            z = self.activation(z)
        z = self.linear_out(z)
        if self.layer_norm:
            z = self.ln_out(z)
        z = z.tanh() # authors say that a final tanh non-linearity gives the best results
        z = z.view(z.size(0), self.hidden_channels, self.input_channels) # batch is first input
        return z
    
    def reset_nfe(self):
        self.nfe = 0

class NeuralCDE(torch.nn.Module):
    def __init__(self, vector_field, input_channels, hidden_channels, output_channels, rtol=1e-4, atol=1e-6, layer_norm=False, seminorm=False):
        super(NeuralCDE, self).__init__()
        self.func = vector_field
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
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

    def forward(self, coeffs, times, interpolation_method):
        X, cdeint_options = build_data_path(coeffs, times, interpolation_method)
        z0 = self.initial(X.evaluate(X.interval[0])) # initial hidden state must be a function of the first observation

        adjoint_options = cdeint_options.copy()
        if self.seminorm:
            adjoint_options['norm'] = make_norm(z0)
       
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

class RNN(torch.nn.Module): # Thank you Python Engineer! https://www.youtube.com/watch?v=0_PgWWmauHk
    def __init__(self, input_channels, hidden_channels, num_layers, output_channels):
        super(RNN, self).__init__()
        self.num_layers = num_layers # how many RNN layers stacked together
        self.hidden_channels = hidden_channels
        self.rnn = torch.nn.RNN(input_channels, hidden_channels, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)
    
    def forward(self, x, times, interpolation_method):
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
    
    def forward(self, x, times, interpolation_method):
        # TODO: remove need of passing times and interpolation methods, they are meant for ncde model!
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
    
    def forward(self, x, times, interpolation_method):
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
    def __init__(self, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(ODERNNFunc, self).__init__()
        self.nfe = 0 # number of function evaluations
        layers = [torch.nn.Linear(hidden_channels, hidden_hidden_channels)]
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_channels))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, t, z):
        self.nfe += 1
        return self.sequential(z)
    
    def reset_nfe(self):
        self.nfe = 0

class ODERNN(torch.nn.Module): # From Kidger 2020
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, output_channels, rtol=1e-4, atol=1e-5):
        super(ODERNN, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.rtol = rtol
        self.atol = atol
        self.func = ODERNNFunc(hidden_channels, hidden_hidden_channels, num_hidden_layers)
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
        out = torchdiffeq.odeint_adjoint(func=self.func, y0=h, t=t, method='rk4', rtol=self.rtol, atol=self.atol)
        return out[1]

    def forward(self, x, times, interpolation_method):
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

def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def make_norm(state): # From https://github.com/JoaquinGajardo/FasterNeuralDiffEq/blob/master/models/common.py
    state_size = state.numel()
    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))
    return norm

def init_network_weights(net, std = 0.1): # From Nando. Not used, Pytorch default initialization of layers seems alright.
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=std)
            torch.nn.init.constant_(m.bias, val=0)

def count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)

def add_weight_regularisation(loss_fn, model, scaling=0.01):
    """ Adds L2-regularisation to the loss function """
    def new_loss_fn(true_label, label_predictions):
        total_loss = loss_fn(true_label, label_predictions)
        for parameter in model.parameters():
            if parameter.requires_grad:
                total_loss += scaling * parameter.norm()
        return total_loss
    return new_loss_fn

def compute_multiclass_cross_entropy(true_label, label_predictions): # From Nando
    eps = 1e-10
    ce_loss = -(true_label * torch.log(label_predictions + eps)).sum(dim=1).mean()  # because torch.nn.functional.cross_entropy -> multi-target not supported
    return ce_loss

def compute_total_matches(labels, predictions):
    thresholded_pred = torch.argmax(predictions, dim=1)
    labels_max_indices = torch.argmax(labels, dim=1)
    prediction_matches = (thresholded_pred == labels_max_indices).to(predictions.dtype).sum()
    return prediction_matches

def compute_f1_score(labels, predictions):
    thresholded_pred = torch.argmax(predictions, dim=1).detach().cpu()
    labels_max_indices = torch.argmax(labels, dim=1).detach().cpu()
    f1_score = sklearn.metrics.f1_score(labels_max_indices, thresholded_pred, average=None) # returns per-class F1 scores, taking the mean of this is eq. to average='macro'
    return f1_score

def evaluate_metrics(dataloader, model, times, interpolation_method, device, loss_fn=compute_multiclass_cross_entropy):
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
            pred_y = model(batch_data, times, interpolation_method).squeeze(-1)

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
        
        # Return metrics
        metrics = dict(dataset_size=total_dataset_size, loss=round(total_loss.item(), 5), accuracy=round(total_accuracy.item(), 5), f1_score=round(total_f1_score.mean().item(), 5), perclass_f1= total_f1_score.round(5))
        return metrics

def grid_search_runs(args, repeats=1):
    ''' Makes a grid with every combination of the arguments that are passed as a list
        and one run for each combination.'''
    def combinations(*args):
        combinations_list = []
        for combination in product(*args):
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
    parser.add_argument("--reduced", default=False, action="store_true", help='Use only a fraction of the dataset features (for Crops dataset) [default=%(default)s].')
    parser.add_argument("--time_default", default=False, action="store_true", help='To not pass the original dataset timestamps to the interpolation and model and use the default equally spaced time array of torchde interpolation methods [default=%(default)s].')
    parser.add_argument("--interpol_method", type=str, default='cubic', choices=['cubic', 'linear', 'rectilinear'], help='Interpolation method to use for creating continous data path X [default=%(default)s].')
    parser.add_argument("--ntrain", type=int, default=None, help='Number of train samples [default=%(default)s].')
    parser.add_argument("--nval", type=int, default=None, help='Number of validation samples [default=%(default)s].')
    parser.add_argument("--max_epochs", type=int, default=20, help='Maximum number of epochs [default=%(default)s].')
    parser.add_argument("--lr", type=float, default=0.001, nargs='*', help='Optimizer initial learning rate [default=%(default)s].')
    parser.add_argument("--BS", type=int, default=600, nargs='*', help='Batch size for train, validation and test sets [default=%(default)s].')
    parser.add_argument("--num_workers", type=int, default=0, help='Num workers for dataloaders [default=%(default)s].')
    parser.add_argument("--HC", type=int, default=80, nargs='*', help='Hidden channels. Size of the hidden state in NCDE or RNN models [default=%(default)s].')
    parser.add_argument("--HL", type=int, default=1, nargs='*', help='Number of hidden layers in the vector field or of RNN layers if an RNN model is selected [default=%(default)s].')
    parser.add_argument("--HU", type=int, default=128, nargs='*', help='Number of hidden units in the vector field [default=%(default)s].')
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'leaky', 'elu', 'tanh'],  help='Intermediate activation function in vector field of NCDE (final one is always tanh) [default=%(default)s].')
    parser.add_argument("--layer_norm", default=False, action="store_true", help='Apply layer norm to before every activation function [default=%(default)s].')
    parser.add_argument("--ES_patience", type=int, default=5, help='Early stopping number of epochs to wait before stopping [default=%(default)s].')
    parser.add_argument("--lr_decay", default=False, action="store_true", help='Add learning rate decay if when no improvement of training accuracy [default=%(default)s].')
    parser.add_argument("--lr_decay_factor", type=float, default=0.5, nargs='*', help='Learning rate decay factor [default=%(default)s].')
    parser.add_argument("--regularization", type=float, default=None, help='If not None adds L2 regularization to the loss function with scaling specified by the argument [default=%(default)s].')
    parser.add_argument("--pin_memory", default=False, action="store_true", help='Pass pin memory option to torch.utils.data.DataLoader [default=%(default)s].')
    parser.add_argument("--save", type=str, default='results', help='Name of new or existing folder where to save results [default=%(default)s].')
    parser.add_argument("--resume", default=None, help='ID of experiment for resuming training. If None runs a new experiment [default=%(default)s].')
    parser.add_argument("--no_logwandb", default=False, action='store_true', help='Log the run in weights and biases [default=%(default)s].')		
    parser.add_argument("--fast_dataloader", default=False, action='store_true', help='Try out fast dataloader (with shuffle=False) [default=%(default)s].')		
    parser.add_argument("--grad_clip", type=float, default=None, help='Max norm to clip gradients to [default=%(default)s].')		
    parser.add_argument("--model", type=str, default='ncde', choices=['ncde', 'odernn', 'rnn', 'gru', 'lstm'], help='Model to use [default=%(default)s].')
    parser.add_argument("--seminorm", default=False, action="store_true", help='If to use seminorm for 2x speed up odeint adaptative solvers [default=%(default)s].')
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
    hidden_hidden_channels = args_dict['HU'] = args_dict['HU'][0] if isinstance(args_dict['HU'], list) else args_dict['HU']
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
        wandb.init(config=args_dict, project='crops_gpu_2.0', save_code=True)
        expID = wandb.run.id
        wandb.run.name = expID
        wandb.run.save()
    else:
        expID = int(random.SystemRandom().random()*1e7)
    print(f'Experiment {expID}')
    print(f'Arguments: \n {args_dict}')

    # Get data and labels
    data = get_data(absolute_data_directory_path=absolute_data_directory_path, use_noskip=noskip, reduced=reduced, ntrain=ntrain_samples, nval=nval_samples, use_model=use_model)
    times = None if time_default and use_model == 'ncde' else data['times']
    coeffs_directory = os.path.join(absolute_data_directory_path, 'interpolation_coefficients')
    if use_model == 'ncde':
        coefficients = get_interpolation_coeffs(coeffs_directory, data, times, interpolation_method=interpolation_method, use_noskip=noskip, reduced=reduced)
        fig = plot_interpolation_path(coefficients, 'train', times, interpolation_method)
        if logwandb:
            wandb.log({'Interpolation sample': fig})

    # Define parametes
    input_channels = data['train_data'].size(2) # 54 features + 1 for time
    output_channels = data['train_labels'].size(1) # 19 field classes
    hidden_channels = hidden_channels  # dimension of the hidden state z
    hidden_hidden_channels = hidden_hidden_channels # vector field network layers size
    num_hidden_layers = num_hidden_layers # vector field network number of layers

    # Use GPU if available
    print('GPU devices available:', torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Define model
    if use_model == 'ncde':
        vector_field = CDEFunc(input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, activation_func, layer_norm=layer_norm)
        model = NeuralCDE(vector_field=vector_field, input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels, layer_norm=layer_norm, seminorm=seminorm, rtol=rtol, atol=atol).to(device)
    elif use_model == 'rnn':
        model = RNN(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_hidden_layers, output_channels=output_channels).to(device)
    elif use_model == 'gru':
        model = GRU(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_hidden_layers, output_channels=output_channels).to(device)
    elif use_model == 'lstm':
        model = LSTM(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_hidden_layers, output_channels=output_channels).to(device)
    elif use_model =='odernn':
        model = ODERNN(input_channels=input_channels, hidden_channels=hidden_channels, hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers, output_channels=output_channels, rtol=rtol, atol=atol).to(device)
    
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
    if use_model == 'ncde':
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
    output = []
    config = dict(script_name=script_name, expID=expID, num_parameters=count_parameters(model))
    if logwandb:
        wandb.config.update(config)
    config.update(args_dict)
    output.append({'config': config})

    # Training and validation loop
    best_val_f1 = 0
    best_val_f1_epoch = 0
    current_lr = learning_rate 
    is_ode = use_model == 'ncde' or use_model == 'odernn'

    print('Learning...')
    if not time_default: times = times.to(device) # Shouldn't give problems because times is None otherwise.
    start = time.time()
    pbar = tqdm.tqdm(range(1, num_epochs + 1))
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
            pred_y = model(batch_data, times, interpolation_method).squeeze(-1)
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
                    wandb.log({f'gradient/{name}': wandb.Histogram(parameter.grad.detach().cpu().numpy())}) 

        # Compute traininig and validation metrics for the epoch
        model.eval()
        train_metrics = evaluate_metrics(train_dataloader, model, times, interpolation_method, device, loss_fn=loss_fn)	# TODO: check that is doesn't slow down everything too much.	
        val_metrics = evaluate_metrics(val_dataloader, model, times, interpolation_method, device, loss_fn=loss_fn) # TODO: check that is doesn't slow down everything too much.
        epoch_time = round(time.time() - start_epoch, 1)
        total_nfes = epoch_nfes + epoch_bnfes
        pbar.write(f'Epoch: {epoch}, lr={current_lr}, epoch time: {epoch_time}s, nfes: {epoch_nfes}, bnfes: {epoch_bnfes}, total nfes: {total_nfes}  || Training loss: {train_metrics["loss"]: 0.5f}  | Training accuracy: {train_metrics["accuracy"]: 0.5f} | Training F1-score: {train_metrics["f1_score"]: 0.5f}  || Validation loss: {val_metrics["loss"]: 0.5f}  | Validation accuracy: {val_metrics["accuracy"]: 0.5f} | Validation F1-score: {val_metrics["f1_score"]: 0.5f}')
        output.append(dict(epoch=epoch, lr=current_lr, epoch_time=epoch_time, nfes=epoch_nfes, bnfes=epoch_bnfes, total_nfes=total_nfes, train_metrics=train_metrics, val_metrics=val_metrics))

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
            best_model = copy.deepcopy(model)
            
            # Save checkpoint of the model # TODO: check that is doesn't slow down everything too much.
            checkpoint_path = os.path.join(checkpoints_path, f'exp_{expID}_checkpoint.pt')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_metrics["loss"],
            'args_dict': args_dict
            }, checkpoint_path)
            
        elif epoch >= best_val_f1_epoch + early_stopping_patience:
            pbar.write(f'Breaking because of no improvement in validation F1-score after {early_stopping_patience} epochs.')
            break

    train_time = round(time.time() - start, 1)
    print(f'Train time: {train_time} seconds')
    if logwandb:
        wandb.log({'train time': train_time})

    # Evaluate performance on test dataset
    print('Testing...')
    #model = best_model
    #del best_model # to release memory
    model.eval()
    test_metrics_best_model = evaluate_metrics(test_dataloader, best_model, times, interpolation_method, device, loss_fn=loss_fn)
    test_metrics_last_model = evaluate_metrics(test_dataloader, model, times, interpolation_method, device, loss_fn=loss_fn)
    
    # Check that best model is indeed better than the last one
    if test_metrics_best_model["f1_score"] >= test_metrics_last_model["f1_score"] and test_metrics_best_model["accuracy"] >= test_metrics_last_model["accuracy"]:
        test_acc = test_metrics_best_model["accuracy"]
        test_f1 = test_metrics_best_model["f1_score"]
        best_epoch = best_val_f1_epoch
    else: # save checkpoint of last model and save these metrics instead 
        test_acc = test_metrics_last_model["accuracy"]
        test_f1 = test_metrics_last_model["f1_score"]
        best_epoch = epoch
        checkpoint_path = os.path.join(checkpoints_path, f'exp_{expID}_checkpoint.pt')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_metrics["loss"], 'args_dict': args_dict}, checkpoint_path)

    print(f'Test accuracy best model (epoch {best_epoch}): {test_acc}, Test f1 score best model: {test_f1}')

    # Log test metrics
    if logwandb:
        wandb.log({'test accuracy (best)': test_acc, 'test F1-score (best)': test_f1, 'best epoch': best_epoch})

    output.append(dict(best_epoch=best_epoch, test_accuracy=test_acc, test_f1_score=test_f1, train_time=train_time))
    
    # Write results to a text file
    print('Output to text file:\n', output)
    noskip = 'noskip' if noskip else ''
    red = 'red' if reduced else ''
    timing = 'eqspaced' if times is None or interpolation_method == 'rectilinear' else 'irrspaced'
    results_path = os.path.join(here, results_folder)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    n = 0
    while glob.glob(os.path.join(results_path, f'*results{n}*_{batch_size}BS_{learning_rate}lr_{lr_decay_factor}lrdecay_{hidden_channels}HC_{num_hidden_layers}HL_{hidden_hidden_channels}HU_{interpolation_method}_{timing}{noskip}{red}.txt')):
        n += 1
    f = open(os.path.join(results_path, f'results{n}_{expID}_{batch_size}BS_{learning_rate}lr_{lr_decay_factor}lrdecay_{hidden_channels}HC_{num_hidden_layers}HL_{hidden_hidden_channels}HU_{interpolation_method}_{timing}{noskip}{red}.txt'), 'w')
    for line in output:
        f.write(str(line) +'\n')
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    main(args=parse_args())
    print('Run time', round(time.time() - start_time, 2), 'seconds')

    
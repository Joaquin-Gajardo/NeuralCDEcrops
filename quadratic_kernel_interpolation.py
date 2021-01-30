"""
Created on Fri Jan 23 2021
@author: Stefano D'Aronco

Remarks:
I wrote the gaussian process-like interpolation method (it is more correct to call it kernel regression since we do not use the estimated variance…):
- It works as the  cubic and linear basically, there is the function to compute the coefficients and the class for interpolation.
- The hyperparameters of the kernel regression are reported at the top of the python file, you can play with those (I set them to some reasonable values, but who knows…)
- NaNs can be in any place so I would NOT fill the beginning and the end of the squence with valid data.
- It is mandatory to pass the time vector but you can easily adapt the code to make it optional. in this case, if the time scale changes the hyperparameter l_kernel must change accordingly.
- It seems to work but there might be some bugs, so double check
- It is still bloody slow, maybe a bit faster but…

"""

import torch
import torchcde

sigma_kernel = 0.25 # the larger this value is the more it allows the fitted function to take large values
l_kernel = 0.05 # this is the width of the kernel and decides how much the function correlates with two nearby points (see https://stats.stackexchange.com/questions/445484/does-length-scale-of-the-kernel-in-gaussian-process-directly-relates-to-correlat )
sigma_noise = 0.05 # this sets the noise std for the valid measurements, the smaller this value, the closer the curve will be to the "control points"


def quadratic_kernel_interpolation_coeffs(x, t):
    """ creates kernel matrix for gaussian process-like interpolation using an exponential quadratic kernel

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        t: one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]). If you are using neural CDEs then you **do not need to use this
            argument**. See the Further Documentation in README.md.
        sigma_kernel: the larger this value is the more it allows the fitted function to take large values
        l_kernel: this is the width of the kernel and decides how much the function correlates with two nearby points (see https://stats.stackexchange.com/questions/445484/does-length-scale-of-the-kernel-in-gaussian-process-directly-relates-to-correlat )
        sigma_noise:  this sets the noise std for the valid measurements, the smaller this value, the closer the curve will be to the "control points"


    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.


    Returns:
        A tensor, which should in turn be passed to `QuadraticKernelInterpolation`.

    """
 
    # x shape  B x T x F
    B,T,F = x.shape
    # t shape T
    #print(x.shape)

    x = x.transpose(1,2)
    # x shape  B x F x T
    x = x.reshape((B*F,T))
    valid_mask = torch.logical_not(x.isnan())

    x_clean = x.clone()
    x_clean[torch.logical_not(valid_mask)] = 0.
    
    noise = torch.ones_like(valid_mask).float()
    noise[valid_mask] = sigma_noise**2
    noise[torch.logical_not(valid_mask)] = 100.**2 # super high variance means no reliable observation
    del valid_mask
    noise_matrix = torch.diag_embed(noise)
    # K shape B*F x T x T

    t = t.unsqueeze(dim=1)
    K = sigma_kernel**2 * torch.exp(-((t.transpose(0,1) - t )/(2*l_kernel))**2)
    K = K.unsqueeze(0)
    # K shape 1 x T x T
    assert(not K.isnan().any())

    #print("K shape: {}".format(K.shape))
    #print("noise shape: {}".format(noise_matrix.shape))

    noise_matrix.add_(K.expand_as(noise_matrix))
    K = noise_matrix
    del noise_matrix
    assert(not K.isnan().any())
    #del noise_matrix

    ### to many smaples we need to do this in batches
    L = 10
    idx = torch.linspace(0,K.shape[0],L)
    idx = idx.long()
    for i in range(0,L-1):
        K[idx[i]:idx[i+1],:,:] = torch.inverse(K[idx[i]:idx[i+1],:,:])
    assert(not K.isnan().any())

    alpha = torch.bmm(K,x_clean.unsqueeze(2))
    assert(not alpha.isnan().any())
    del K
    # alpha shape B*F x T
    alpha = alpha.reshape(B,F,T)
    alpha = alpha.transpose(1,2)

    return alpha


class QuadraticKernelInterpolation(torch.nn.Module):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, coeffs, t, **kwargs):
        """
        Arguments:
            coeffs: As returned by linear_interpolation_coeffs.
            t: As passed to linear_interpolation_coeffs.
        """
        super(QuadraticKernelInterpolation, self).__init__(**kwargs)

        #self._t = t
        #self._coeffs = coeffs

        torchcde.misc.register_computed_parameter(self, '_t', t)
        torchcde.misc.register_computed_parameter(self, '_coeffs', coeffs)
        #misc.register_computed_parameter(self, '_derivs', derivs)
        #self._reparameterise = reparameterise

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def evaluate(self, t):

        squeeze = False
        if len(t.shape) == 0:
            squeeze = True
            t = t.unsqueeze(0)

        k_star = sigma_kernel**2 * torch.exp(-((self._t.unsqueeze(1).transpose(0,1) - t.unsqueeze(1))/(2*l_kernel))**2)
        k_star = k_star.unsqueeze(0).unsqueeze(3)
        # k_star shape 1 x T_test x T x 1
        # _coeffs shape B x (1 x) T x F

        f_star = torch.sum(self._coeffs.unsqueeze(1) * k_star, dim=2)
        
        # f_star shape B x T_test x F

        if squeeze:
            f_star = f_star.squeeze(1)

        return f_star

    def derivative(self, t):

        squeeze = False
        if len(t.shape) == 0:
            squeeze = True
            t = t.unsqueeze(0)

        k_star_der = ((self._t.unsqueeze(1).transpose(0,1) - t.unsqueeze(1))/(2*l_kernel)) * sigma_kernel**2 * torch.exp(-((self._t.unsqueeze(1).transpose(0,1) - t.unsqueeze(1))/(2*l_kernel))**2)
        k_star_der = k_star_der.unsqueeze(0).unsqueeze(3)

        f_star_der = torch.sum(self._coeffs.unsqueeze(1) * k_star_der,dim=2)
        
        if squeeze:
            f_star_der = f_star_der.squeeze(1)

        return f_star_der

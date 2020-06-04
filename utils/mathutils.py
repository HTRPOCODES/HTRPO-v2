import numpy as np

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def popart_update(weight, bias, mu_old, mu_new, sigma_old, sigma_new):
    """
    :param weight: Nout x Nin Tensor.
    :param bias:  Nout x 1 Tensor
    :param mu_old: Nout x 1 Tensor
    :param mu_new: Nout x 1 Tensor
    :param sigma_old: Nout x 1 Tensor
    :param sigma_new: Nout x 1 Tensor
    :return:
    """
    return 1 / sigma_new * sigma_old * weight, 1 / sigma_new * (sigma_old * bias + mu_old - mu_new)

def half_uniform_to_uniform(uniform_range, weight, axis = 0):
    """
    :param uniform_range: a N-d vector to define the range of the uniform distribution
    :param weight: a scalar to define the weight of the two parts.
    :param axis: which axis should be split into two parts.
    :return: a transformer
    """

    pass
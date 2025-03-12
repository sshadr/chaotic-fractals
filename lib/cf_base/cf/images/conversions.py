import numpy as np
import torch


def image_to_tensor(img, *, normalize=False, batch_dim=True, to_cuda=False):
    """
    Converts a numpy array image to a torch tensor image.

    Parameters
    ----------
    img : np.ndarray
        The image as a numpy array of shape (height, width, channels).
    normalize : bool
        If true, transform the data range from [0, 1] to [-1, 1].
    batch_dim : bool
        If true, add a batch dimension as the first dimension. The resulting tensor will then be 4-dimensional.
    to_cuda : bool

    Returns
    -------
    tensor: torch.Tensor
        The image as a torch tensor of shape (channels, height, width) or (1, channels, height, width).
    """

    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    if to_cuda:
        tensor = tensor.cuda()
    if normalize:
        tensor = tensor * 2 - 1
    if batch_dim:
        tensor = tensor[None]
    return tensor


def tensor_to_image(tensor, *, denormalize=False):
    """
    Converts a torch tensor image to a numpy array image.

    Parameters
    ----------
    tensor : torch.Tensor
        The image as a torch tensor of shape (channels, height, width) or (1, channels, height, width).
    denormalize : bool
        If true, transform the data range from [-1, 1] to [0, 1].

    Returns
    -------
    img : np.ndarray
        The image as a numpy array of shape (height, width, channels).
    """

    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("If the image tensor has a batch dimension, it must have length 1.")
        tensor = tensor[0]
    if denormalize:
        tensor = (tensor + 1) * 0.5
    img = tensor.numpy(force=True)
    return np.transpose(img, (1, 2, 0))

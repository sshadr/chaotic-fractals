import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.optim_utils import normalize_tensor, gauss_filter

def compute_mip_loss(downgts, downimgs, num_levels, filters):
        # inputs are full scale images
        p_loss = 0
        mip_filter = filters[0]
        
        for i in range(num_levels+1):
            if i > 0:
                downgts = normalize_tensor(mip_filter(downgts))
                downimgs = normalize_tensor(mip_filter(downimgs))
                
            p_loss += (recon_loss(downimgs, downgts))
        return p_loss 

def constrain_bias(bias, min=-1, max=1):
    """
    weight decay on bias
    """
    diff_tensor = (bias)
    squared_diff_tensor = diff_tensor ** 2
    total_loss = torch.sum(squared_diff_tensor)
    return total_loss

def lipschitz_constraint_sigma(sigmas_list, threshold=1.0):
    """
    weight decay on both the singular values
    """
    singular_values_tensor = torch.cat(sigmas_list, dim=0)

    diff_tensor = (singular_values_tensor)
    squared_diff_tensor = diff_tensor ** 2
    total_loss = torch.sum(squared_diff_tensor)

    return total_loss

def constrain_condition_number(sigmas_list):
    c_numbers = []
    for i in range(len(sigmas_list)):
        sigmas = sigmas_list[i]
        condition_number = (sigmas.max() + torch.tensor(1))/(sigmas.min() + torch.tensor(1)) # + torch.tensor(0.0001))
        
        c_numbers.append(condition_number)
    
    c_tensor = torch.stack(c_numbers)

    # minimize this condition number(so its technically allowed to go to 1) but use a low regularization constant
    condition_number_loss = torch.mean((c_tensor) ** 2)
    
    return condition_number_loss


def recon_loss(img, gt):
    l = torch.mean(torch.square(img - gt))
    return l


def compute_moments(image_tensor):
    # Assuming image_tensor is a 4D tensor with shape (batch_size, channels, height, width)
    if image_tensor.dim() == 4 and image_tensor.size(1) == 3:
        image_tensor = image_tensor[:, 0, :, :]
    
    # Get the shape of the image
    height, width = image_tensor.size(1), image_tensor.size(2)
    
    # Create coordinate arrays
    x_coords = torch.arange(width, device=image_tensor.device).repeat(height, 1).flatten() / width
    y_coords = torch.arange(height, device=image_tensor.device).repeat(width, 1).t().flatten() / height
    
    # Flatten the pixel values
    pixel_values = image_tensor.flatten()
    
    # Compute the weighted mean of the x and y coordinates
    total_intensity = torch.sum(pixel_values)
    mean_x = torch.sum(x_coords * pixel_values) / total_intensity
    mean_y = torch.sum(y_coords * pixel_values) / total_intensity
    
    # Compute the variance (second central moment)
    var_x = torch.sum(pixel_values * (x_coords - mean_x) ** 2) / total_intensity
    var_y = torch.sum(pixel_values * (y_coords - mean_y) ** 2) / total_intensity
    
    # Compute the skewness (third standardized moment)
    skew_x = (torch.sum(pixel_values * (x_coords - mean_x) ** 3) / total_intensity) / (var_x ** 1.5)
    skew_y = (torch.sum(pixel_values * (y_coords - mean_y) ** 3) / total_intensity) / (var_y ** 1.5)
    
    # Compute the kurtosis (fourth standardized moment)
    kurt_x = (torch.sum(pixel_values * (x_coords - mean_x) ** 4) / total_intensity) / (var_x ** 2)
    kurt_y = (torch.sum(pixel_values * (y_coords - mean_y) ** 4) / total_intensity) / (var_y ** 2)

    m5_x = (torch.sum(pixel_values * (x_coords - mean_x) ** 5) / total_intensity) / (var_x ** 2.5)
    m5_y = (torch.sum(pixel_values * (y_coords - mean_y) ** 5) / total_intensity) / (var_y ** 2.5)

    m6_x = (torch.sum(pixel_values * (x_coords - mean_x) ** 6) / total_intensity) / (var_x ** 3)
    m6_y = (torch.sum(pixel_values * (y_coords - mean_y) ** 6) / total_intensity) / (var_y ** 3)


    moments = {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "var_x": var_x,
        "var_y": var_y,
        "skew_x": skew_x,
        "skew_y": skew_y,
        "kurt_x": kurt_x,
        "kurt_y": kurt_y,
        # "m5x": m5_x,
        # "m5y": m5_y,
        # "m6x": m6_x,
        # "m6y": m6_y
    }
    
    return moments

def moments_loss(image_tensor1, image_tensor2):
    moments1 = compute_moments(image_tensor1)
    moments2 = compute_moments(image_tensor2)
    
    loss = 0
    for key in moments1:
        loss += torch.square(moments1[key] - moments2[key])
    
    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
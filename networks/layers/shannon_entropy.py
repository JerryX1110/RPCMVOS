import torch
import numpy as np

def normalize(image, MIN_BOUND, MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    reverse_image = 1 - image
    return reverse_image

def cal_shannon_entropy(preds): 
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True) 
    uncertainty_norm = normalize(uncertainty, 0, np.log(2)) * 7
    return uncertainty,uncertainty_norm


def normalize_train(image, MIN_BOUND, MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return image

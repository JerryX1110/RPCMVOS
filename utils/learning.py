import math


def adjust_learning_rate(optimizer, base_lr, p, itr, max_itr, warm_up_steps=1000, is_cosine_decay=False, min_lr=1e-5):
    
    if itr < warm_up_steps:
        now_lr = base_lr * itr / warm_up_steps
    else:
        itr = itr - warm_up_steps
        max_itr = max_itr - warm_up_steps
        if is_cosine_decay:
            now_lr = base_lr * (math.cos(math.pi * itr / (max_itr + 1)) + 1.) * 0.5
        else:
            now_lr = base_lr * (1 - itr / (max_itr + 1)) ** p

    if now_lr < min_lr:
        now_lr = min_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = now_lr
    return now_lr


def get_trainable_params(model, base_lr, weight_decay, beta_wd=True):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        wd = weight_decay
        if 'beta' in key:
            if not beta_wd:
                wd = 0.
        params += [{"params": [value], "lr": base_lr, "weight_decay": wd}]
    return params